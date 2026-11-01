#include "dynamatic/Support/ConstraintProgramming/ConstraintProgramming.h"
#include "dynamatic/Support/LLVM.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>

#ifdef DYNAMATIC_ENABLE_HIGHS
#include "Highs.h"
#endif // DYNAMATIC_ENABLE_HIGHS

using namespace dynamatic;

// -------------------------------------------------------------
// Utility functions
// -------------------------------------------------------------

// HACK: since we don't assign any names to the MILP model, we use this global
// variable to distingush between different MILP models.
static unsigned int modelCount = 0;

namespace dynamatic {
namespace detail {

#ifdef DYNAMATIC_ENABLE_CBC

// Solution parser:
//
// Example:
// Optimal - objective value 9900.00000000
//       0 numExec_times_sArc_0_1               0                       1
//       1 numExec_times_sArc_1_2               0                       1
//       2 numExec_times_sArc_2_2            9900                       1
//       3 numExec_times_sArc_2_3               0                       1
// NOTE: sometimes Cbc will not disable the variable whose solution is 0.0.
mlir::LogicalResult CbcSoluParser::parseSolverOutput(StringRef soluFileName) {
  auto bufferOrErr = llvm::MemoryBuffer::getFile(soluFileName);
  if (!bufferOrErr)
    return mlir::failure();

  llvm::line_iterator it(*bufferOrErr->get(), /*SkipBlanks=*/true);

  // ---- First line: status + objective value ----
  // NOTE: The default constructor of llvm::line_iterator is the "end" iterator
  if (it == llvm::line_iterator())
    return failure();

  {
    // Example:
    // "Optimal - objective value 999.00000000"
    StringRef line = *it;

    if (line.starts_with("Optimal")) {
      status = CPSolver::Status::OPTIMAL;
    } else if (line.starts_with("Infeasible")) {
      status = CPSolver::Status::INFEASIBLE;
    } else if (line.starts_with("Unbounded")) {
      status = CPSolver::Status::UNBOUNDED;
    } else if (line.starts_with("Stopped on time")) {
      status = CPSolver::Status::NONOPTIMAL;
    } else {
      status = CPSolver::Status::UNKNOWN;
    }

    // Extract objective value
    auto pos = line.find("objective value");
    if (pos != StringRef::npos) {
      StringRef valueStr =
          line.drop_front(pos + strlen("objective value")).trim();
      valueStr.getAsDouble(objectiveValue);
    }
  }
  ++it;
  // ---- Remaining lines: variable assignments ----
  for (; it != llvm::line_iterator(); ++it) {
    // Example:
    // "1 numExec_times_sArc_1_1 999 1"
    std::stringstream ss(it->str());
    int index;
    std::string varName;
    double value;
    ss >> index >> varName >> value;
    if (ss.fail())
      continue;
    results[varName] = value;
  }

  return success();
}
#endif // DYNAMATIC_ENABLE_CBC

// Avoid cyclic dependency in the overloading.
static LinExpr addVarConstImpl(const CPVar &left, double right) {
  // REMARK:
  // this is a deep copy of "left"
  LinExpr newExpr = left;
  // REMARK:
  // std::map makes this safe when var does not exist in "right" at the
  // first place by calling the default constructor, i.e., setting terms[right]
  // = 0.0.
  newExpr.constant = right;
  return newExpr;
}

// Avoid cyclic dependency in the overloading.
static LinExpr subVarConstImpl(const CPVar &left, double right) {

  // REMARK:
  // this is a deep copy of "left"
  LinExpr newExpr = left;
  // REMARK:
  // std::map makes this safe when var does not exist in "right" at the
  // first place by calling the default constructor, i.e., setting terms[right]
  // = 0.0.
  newExpr.constant = -right;
  return newExpr;
}

static LinExpr mulVarConstImpl(const CPVar &left, double right) {
  LinExpr newExpr(left);
  newExpr.terms[left] *= right;
  newExpr.constant *= right;
  return newExpr;
}
} // namespace detail

// -------------------------------------------------------------
// Operator overloading implementation for the LP DSL
// ------------------------------------------------------------

bool CPVar::operator<(const CPVar &other) const noexcept {
  return this->impl->name < other.impl->name;
}

std::string CPVar::getName() const { return this->impl->name; }

LinExpr operator+(const CPVar &left, double right) {
  return detail::addVarConstImpl(left, right);
}

LinExpr operator+(double left, const CPVar &right) {
  return detail::addVarConstImpl(right, left);
}

LinExpr operator-(const CPVar &left, double right) {
  return detail::subVarConstImpl(left, right);
}

LinExpr operator-(double left, const CPVar &right) {
  return -detail::subVarConstImpl(right, left);
}

LinExpr operator+(const CPVar &left, const CPVar &right) {
  // REMARK:
  // this is a deep copy of "left"
  LinExpr newExpr = left;
  // REMARK:
  // std::map makes this safe when var does not exist in "right" at the
  // first place by calling the default constructor, i.e., setting terms[right]
  // = 0.0.
  newExpr.terms[right] += 1.0;
  return newExpr;
}

LinExpr operator-(const CPVar &left, const CPVar &right) {
  // REMARK:
  // this is a deep copy of "left"
  LinExpr newExpr = left;
  // REMARK:
  // std::map makes this safe when var does not exist in "right" at the
  // first place by calling the default constructor, i.e., setting terms[right]
  // = 0.0.
  newExpr.terms[right] -= 1.0;
  return newExpr;
}

LinExpr operator+(const LinExpr &left, double right) {
  // REMARK:
  // this is a deep copy of "left"
  LinExpr newExpr = left;
  newExpr.constant += right;
  return newExpr;
}

LinExpr operator+(double left, const LinExpr &right) {
  // REMARK:
  // this is a deep copy of "left"
  return (right + left);
}

LinExpr operator+(const LinExpr &left, const LinExpr &right) {
  // REMARK:
  // this is a deep copy of "left"
  LinExpr newExpr = left;
  for (auto &[var, coeff] : right.terms) {
    // REMARK:
    // std::map makes this safe when var does not exist in "left" at the
    // first place by calling the default constructor, i.e., setting terms[var]
    // = 0.0. This happens when "left" and "right" have different variables.
    newExpr.terms[var] += coeff;
  }
  newExpr.constant += right.constant;
  return newExpr;
}

void operator+=(LinExpr &left, const LinExpr &right) {
  for (auto &[var, coeff] : right.terms) {
    left.terms[var] += coeff;
  }
  left.constant += right.constant;
}

LinExpr operator-(const LinExpr &left, const LinExpr &right) {
  LinExpr newExpr = left;
  for (auto &[var, coeff] : right.terms) {
    newExpr.terms[var] -= coeff;
  }
  newExpr.constant -= right.constant;
  return newExpr;
}

void operator-=(LinExpr &left, const LinExpr &right) {
  for (auto &[var, coeff] : right.terms) {
    left.terms[var] -= coeff;
  }
  left.constant -= right.constant;
}

LinExpr operator*(double c, const CPVar &v) {
  return detail::mulVarConstImpl(v, c);
}

LinExpr operator*(const CPVar &v, double c) {
  return detail::mulVarConstImpl(v, c);
}

LinExpr operator*(double c, const LinExpr &expr) {
  LinExpr newExpr(expr);
  for (auto &[var, coeff] : newExpr.terms) {
    newExpr.terms[var] = c * coeff;
  }
  newExpr.constant *= c;
  return newExpr;
}
LinExpr operator*(const LinExpr &expr, double c) { return c * expr; }

QuadExpr operator*(const LinExpr &lhs, const LinExpr &rhs) {
  QuadExpr e;
  // Quadratic terms:
  for (auto &[lhsTerm, lhsCoeff] : lhs.terms) {
    for (auto &[rhsTerm, rhsCoeff] : rhs.terms) {
      e.quadTerms[makeSortedPair(lhsTerm, rhsTerm)] += lhsCoeff * rhsCoeff;
    }
  }
  // Linear terms:
  for (auto &[v, coeff] : lhs.terms) {
    e.linexpr.terms[v] += coeff * rhs.constant;
  }
  for (auto &[v, coeff] : rhs.terms) {
    e.linexpr.terms[v] += coeff * lhs.constant;
  }
  // Constant term:
  e.linexpr.constant = lhs.constant * rhs.constant;
  return e;
}

QuadExpr operator*(double lhs, const QuadExpr &rhs) {
  QuadExpr e = rhs;
  // Quadratic terms:
  for (auto &[quadTerm, coeff] : e.quadTerms) {
    e.quadTerms[quadTerm] = coeff * lhs;
  }
  for (auto &[linTerm, coeff] : e.linexpr.terms) {
    e.linexpr.terms[linTerm] = coeff * lhs;
  }
  return e;
}

QuadExpr operator+(const QuadExpr &lhs, const QuadExpr &rhs) {
  QuadExpr e = lhs;
  // Quadratic terms:
  for (auto &[quadTerm, coeff] : rhs.quadTerms) {
    e.quadTerms[quadTerm] += coeff;
  }
  // Linear and constant terms:
  e.linexpr = e.linexpr + rhs.linexpr;
  return e;
}

QuadExpr operator-(const QuadExpr &lhs, const QuadExpr &rhs) {
  QuadExpr e = lhs;
  // Quadratic terms:
  for (auto &[quadTerm, coeff] : rhs.quadTerms) {
    e.quadTerms[quadTerm] -= coeff;
  }
  // Linear and constant terms:
  e.linexpr = e.linexpr - rhs.linexpr;
  return e;
}

void operator+=(QuadExpr &lhs, const QuadExpr &rhs) {
  // Quadratic terms:
  for (auto &[quadTerm, coeff] : rhs.quadTerms) {
    lhs.quadTerms[quadTerm] += coeff;
  }
  // Linear and constant terms:
  lhs.linexpr = lhs.linexpr + rhs.linexpr;
}

void operator-=(QuadExpr &lhs, const QuadExpr &rhs) {
  // Quadratic terms:
  for (auto &[quadTerm, coeff] : rhs.quadTerms) {
    lhs.quadTerms[quadTerm] -= coeff;
  }
  // Linear and constant terms:
  lhs.linexpr = lhs.linexpr - rhs.linexpr;
}

// -------------------------------------------------------------
// TempConstr method implementations
// ------------------------------------------------------------

TempConstr operator<=(const QuadExpr &lhs, const QuadExpr &rhs) {
  QuadExpr e = (lhs - rhs);
  TempConstr c;
  c.expr = e;
  c.pred = LE;
  return c;
}

TempConstr operator>=(const QuadExpr &lhs, const QuadExpr &rhs) {
  return (rhs <= lhs);
}

TempConstr operator==(const QuadExpr &lhs, const QuadExpr &rhs) {
  QuadExpr e = (lhs - rhs);
  TempConstr c;
  c.expr = e;
  c.pred = EQ;
  return c;
}

static std::string formatCoeffAndName(double coeff, const CPVar &v) {
  std::stringstream ss;
  if (std::abs(coeff) == 1.0) {
    return v.impl->name;
  }
  ss << std::abs(coeff) << " " + v.impl->name;
  return ss.str();
}

std::string LinExpr::writeLp() const {
  std::stringstream ss;
  ss << std::fixed;
  unsigned count = 0;
  for (auto &[term, coeff] : this->terms) {
    if (coeff == 0.0)
      continue;
    if (count == 0) {
      ss << (coeff > 0 ? "" : "- ") << formatCoeffAndName(coeff, term);
    } else {
      ss << (coeff > 0 ? " + " : " - ") << formatCoeffAndName(coeff, term);
    }
    count += 1;
  }
  if (0.0 != std::abs(constant))
    ss << (this->constant > 0 ? " + " : " - ") << std::abs(this->constant);
  return ss.str();
}

std::string TempConstr::writeLp() const {
  if (!this->expr.quadTerms.empty())
    llvm::report_fatal_error("Quandratic formula is unsupported!");

  LinExpr linexpr = this->expr.linexpr;

  std::stringstream ss;
  ss << std::fixed;
  unsigned count = 0;
  for (auto &[term, coeff] : linexpr.terms) {
    if (coeff == 0.0)
      continue;
    if (count == 0) {
      ss << (coeff > 0 ? "" : "- ") << formatCoeffAndName(coeff, term);
    } else {
      ss << (coeff > 0 ? " + " : " - ") << formatCoeffAndName(coeff, term);
    }
    count += 1;
  }

  if (this->pred == EQ)
    ss << " = ";
  else if (this->pred == LE)
    ss << " <= ";
  else
    llvm::report_fatal_error("unknown predicate");

  // TODO: not sure if we can put the constant on the LHS.
  // Put the constant in the RHS and flip the sign
  ss << (linexpr.constant > 0 ? " -" : "") << std::abs(linexpr.constant);
  return ss.str();
}

// -------------------------------------------------------------
// GurobiSolver method implementations
// ------------------------------------------------------------

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED

GurobiSolver::GurobiSolver(int timeout, int maxThreads)
    : CPSolver(timeout, GUROBI, maxThreads) {
  env = std::make_unique<GRBEnv>(true);

  // Suppress outputs to stdout (clashes with the MLIR output file).
  env->set(GRB_IntParam_OutputFlag, 0);

  // Always use the same random seed to make the solution deterministic.
  env->set(GRB_IntParam::GRB_IntParam_Seed, 0);

  if (timeout > 0) {
    env->set(GRB_DoubleParam_TimeLimit, timeout);
  }

  if (maxThreads > 0) {
    env->set(GRB_IntParam_Threads, maxThreads);
  }

  env->start();

  model = std::make_unique<GRBModel>(*env);
}

void GurobiSolver::addConstr(const TempConstr &constraint,
                             llvm::StringRef constrName) {
  if (!constraint.expr.quadTerms.empty())
    llvm::report_fatal_error(
        "Adding a linear constraint with quadratic terms!");

  GRBLinExpr expr = 0;
  // Linear terms
  for (auto &[name, coeff] : constraint.expr.linexpr.terms) {
    expr += coeff * variables[name];
  }

  // Constant terms
  expr += constraint.expr.linexpr.constant;
  if (constraint.pred == LE)
    model->addConstr(expr <= 0, constrName.str());
  else if (constraint.pred == EQ)
    model->addConstr(expr == 0, constrName.str());
  else
    llvm_unreachable("Unknown predicate!");
}

void GurobiSolver::addQConstr(const TempConstr &constraint,
                              llvm::StringRef constrName) {
  GRBQuadExpr expr = 0;

  // Quadratic terms
  for (auto &[name, coeff] : constraint.expr.quadTerms) {
    expr += coeff * variables[name.first] * variables[name.second];
  }

  // Linear terms
  for (auto &[name, coeff] : constraint.expr.linexpr.terms) {
    expr += coeff * variables[name];
  }

  // Constant terms
  expr += constraint.expr.linexpr.constant;
  if (constraint.pred == LE)
    model->addQConstr(expr <= 0, constrName.str());
  else if (constraint.pred == EQ)
    model->addQConstr(expr == 0, constrName.str());
  else
    llvm_unreachable("Unknown predicate!");
}

CPVar GurobiSolver::addVar(const std::string &name, VarType type,
                           std::optional<double> lb, std::optional<double> ub) {
  auto var = CPVar(name, type, lb, ub);
  return addVar(var);
}

CPVar GurobiSolver::addVar(const CPVar &var) {
  if (names.count(var.impl->name)) {
    llvm::report_fatal_error("Adding variable with duplicated names is not "
                             "permitted! Aborting...");
  }
  double lb = var.impl->lowerBound.value_or(-GRB_INFINITY);
  double ub = var.impl->upperBound.value_or(GRB_INFINITY);
  char type;
  switch (var.impl->type) {
  case REAL:
    type = GRB_CONTINUOUS;
    break;
  case INTEGER:
    type = GRB_INTEGER;
    break;
  case BOOLEAN:
    type = GRB_BINARY;
  }
  variables[var] = model->addVar(lb, ub, 0.0, type, var.impl->name);
  names.insert(var.impl->name);
  return var;
}

void GurobiSolver::setMaximizeObjective(const LinExpr &expr) {
  GRBLinExpr obj = 0;
  for (auto &[name, coeff] : expr.terms) {
    obj += coeff * variables[name];
  }
  obj += expr.constant;
  // NOTE: the constant term can be ignored in the objective
  model->setObjective(obj, GRB_MAXIMIZE);
}

void GurobiSolver::optimize() {
  model->optimize();

  switch (model->get(GRB_IntAttr_Status)) {
  case GRB_OPTIMAL:
    status = OPTIMAL;
    break;
  case GRB_SUBOPTIMAL:
  case GRB_TIME_LIMIT:
    status = NONOPTIMAL;
    break;
  case GRB_UNBOUNDED:
  case GRB_INF_OR_UNBD:
    status = UNBOUNDED;
    break;
  case GRB_INFEASIBLE:
    status = INFEASIBLE;
    break;
  default:
    status = ERROR;
  }
}

void GurobiSolver::writeSol(llvm::StringRef filePath) const {
  if (status != OPTIMAL && status != NONOPTIMAL) {
    llvm::report_fatal_error("Calling writeSol before the model was solved!");
  }

  std::ofstream myfile(filePath.str());
  if (myfile.is_open()) {
    for (auto &[var, _] : variables) {
      myfile << var.impl->name << " = " << getValue(var) << "\n";
    }
  } else {
    llvm::errs() << "Unable to open file: " << filePath << "!\n";
    llvm::report_fatal_error("Unable to open file!");
  }
}

double GurobiSolver::getValue(const CPVar &var) const {
  if (status != OPTIMAL && status != NONOPTIMAL) {
    llvm::errs() << "Solution is not available while retrieving "
                 << var.impl->name << "!\n";
    llvm::report_fatal_error("Cannot retrieve the value of variable!");
  }
  return variables.at(var).get(GRB_DoubleAttr_X);
}

double GurobiSolver::getObjective() const {
  if (status != OPTIMAL && status != NONOPTIMAL) {
    llvm::report_fatal_error("Cannot retrieve the objective because the "
                             "solution is not available!");
  }
  return model->get(GRB_DoubleAttr_ObjVal);
}

#endif // DYNAMATIC_GUROBI_NOT_INSTALLED

// -------------------------------------------------------------
// CbcSolver method implementations
// ------------------------------------------------------------

#ifdef DYNAMATIC_ENABLE_CBC

CPVar CbcSolver::addVar(const CPVar &var) {
  if (names.count(var.impl->name)) {
    llvm::report_fatal_error("Adding variable with duplicated names is not "
                             "permitted! Aborting...");
  }
  names.insert(var.impl->name);
  variables.insert(var);
  return var;
}

CPVar CbcSolver::addVar(const std::string &name, VarType type,
                        std::optional<double> lb, std::optional<double> ub) {
  auto var = CPVar(name, type, lb, ub);
  return addVar(var);
}

double CbcSolver::getValue(const CPVar &var) const {
  // NOTE: Cbc rename the variables if the lp file is malformed or the variable
  // contains invalid char. In this case, the Cbc log file will report that.
  // We abort the MLIR solving if the the Cbc log file reports that some
  // variable names are invalid.
  if (!this->solution.results.count(var.impl->name)) {
    return 0.0;
  }
  return this->solution.results.at(var.impl->name);
}

void CbcSolver::writeSol(llvm::StringRef filePath) const {
  // Update solver with solution for getValue
  std::ofstream myfile(filePath.str());
  if (myfile.is_open()) {
    for (const auto &var : variables) {
      myfile << var.impl->name << " = " << getValue(var) << "\n";
    }

    myfile << this->getObjective() << "\n";
  } else {
    llvm::errs() << "Unable to open file: " << filePath << "!\n";
    llvm::report_fatal_error("Unable to open file!");
  }
}

void CbcSolver::writeLp(llvm::StringRef filepath) const {

  std::error_code ec;
  llvm::raw_fd_ostream os(filepath, ec);

  if (ec) {
    llvm::errs() << "Error opening file: " << ec.message() << "\n";
    return;
  }
  os << "Maximize\n";
  os << this->maxObjective.writeLp() << "\n";

  os << "Subject to\n";
  for (const auto &[name, constr] : constraints) {
    os << constr.writeLp() << "\n";
  }
  os << "\n\n";

  unsigned numBool = 0, numInt = 0;

  for (const auto &v : variables) {
    if (v.impl->type == BOOLEAN)
      ++numBool;
    else if (v.impl->type == INTEGER)
      ++numInt;
    else if (v.impl->type == REAL)
      continue;
    else
      // Shouldn't be caused by invalid user input
      llvm_unreachable("Unknown type");
  }

  // Print the bounds:
  for (const auto &v : variables) {
    if (v.impl->type == BOOLEAN)
      continue;
    if (v.impl->upperBound) {
      os << v.impl->name << " <= " << llvm::format("%.6f", *v.impl->upperBound)
         << "\n";
    }
    if (v.impl->lowerBound) {
      os << v.impl->name << " >= " << llvm::format("%.6f", *v.impl->lowerBound)
         << "\n";
    }
  }

  // Declare the data types
  if (numBool) {
    os << "Binary\n";
    for (const auto &v : variables) {
      if (v.impl->type == BOOLEAN)
        os << v.impl->name << " ";
    }
    os << "\n\n";
  }

  if (numInt) {
    os << "General\n";
    for (const auto &v : variables) {
      if (v.impl->type == INTEGER)
        os << v.impl->name << " ";
    }
    os << "\n\n";
  }

  os << "End\n";
}

static bool containsInvalid(StringRef filePath) {
  // 1. Open the file
  auto bufferOrErr = llvm::MemoryBuffer::getFile(filePath);
  if (std::error_code ec = bufferOrErr.getError())
    return false;

  // 2. Access the buffer as a StringRef
  llvm::StringRef content = bufferOrErr.get()->getBuffer();

  // 3. Search for both variations
  // .contains() is available in newer LLVM; otherwise use .find() != npos
  return content.contains("invalid") || content.contains("Invalid");
}

void CbcSolver::optimize() {

  // To make sure that parallel running MILP instances will generate different
  // files, we append the name with the ID of the current thread.
  std::stringstream ss;
  ss << std::this_thread::get_id();
  uint64_t id = std::stoull(ss.str());

  std::string lpFile = llvm::formatv("cbc_model_{0}_{1}.lp", id, modelCount);
  std::string solFile =
      llvm::formatv("cbc_solution_{0}_{1}.sol", id, modelCount);
  std::string redirectFile =
      llvm::formatv("cbc_output_{0}_{1}.log", id, modelCount);
  modelCount += 1;
  this->writeLp(lpFile);

  std::string errMsg;
  bool executionFailed;
  // Find the program in the PATH
  auto programName = llvm::sys::findProgramByName("cbc");
  std::error_code ec = programName.getError();
  if (ec) {
    llvm::report_fatal_error("Could not find cbc in the path!\n");
  }

  std::vector<std::optional<StringRef>> redirects = {std::nullopt, redirectFile,
                                                     std::nullopt};

  int exitCode;

  if (this->timeout > 0)
    exitCode = llvm::sys::ExecuteAndWait(
        *programName,
        {"cbc", lpFile, "sec", std::to_string(timeout), "solve", "solu",
         solFile},
        std::nullopt, redirects, 0, 0, &errMsg, &executionFailed);
  else
    exitCode = llvm::sys::ExecuteAndWait(
        *programName, {"cbc", lpFile, "solve", "solu", solFile}, std::nullopt,
        redirects, 0, 0, &errMsg, &executionFailed);

  if (exitCode != 0) {
    llvm::errs() << "Cbc failed with exit code " << exitCode << "!\n";
    llvm::report_fatal_error("Cbc failed!");
  }

  if (containsInvalid(redirectFile)) {
    std::string errMsg = "The file " + lpFile +
                         " is invalid for Cbc! See the logfile" + redirectFile;
    llvm::report_fatal_error(StringRef(errMsg));
  }

  if (auto res = this->solution.parseSolverOutput(solFile); failed(res))
    llvm::report_fatal_error("The solution is malformed!\n");

  this->status = this->solution.status;
}

double CbcSolver::getObjective() const {
  if (status != OPTIMAL && status != NONOPTIMAL) {
    llvm::report_fatal_error("Cannot retrieve the objective because the "
                             "solution is not available!");
  }

  return this->solution.objectiveValue;
}

void CbcSolver::setMaximizeObjective(const LinExpr &expr) {
  this->maxObjective = expr;
}

void CbcSolver::addConstr(const TempConstr &constraint,
                          llvm::StringRef constrName) {

  this->constraints.emplace_back(constrName, constraint);
}
#endif // DYNAMATIC_ENABLE_CBC

// -------------------------------------------------------------
// Linearization of bilinear terms
// -------------------------------------------------------------

LinExpr linearizeIntegerTimesVar(CPSolver &model, const CPVar &intVar,
                                            unsigned intVarUpperBound,
                                            const CPVar &realVar,
                                            double realVarUpperBound,
                                            llvm::StringRef prefix) {
  std::string name = prefix.str();

  // Smallest number of bits that can represent every value in [0, bound].
  unsigned numBits = 1;
  while ((1u << numBits) - 1u < intVarUpperBound)
    ++numBits;

  LinExpr intExpansion;
  LinExpr product;
  for (unsigned bit = 0; bit < numBits; ++bit) {
    double weight = static_cast<double>(1u << bit);

    std::string bitName = name + "_bit" + std::to_string(bit);
    CPVar bitVar = model.addVar(bitName, BOOLEAN, 0.0, 1.0);
    intExpansion += weight * bitVar;

    // w = bitVar * realVar, expressed with three linear inequalities (plus the
    // variable's own non-negativity bound).
    std::string prodName = name + "_bitprod" + std::to_string(bit);
    CPVar prod = model.addVar(prodName, REAL, 0.0, realVarUpperBound);
    model.addConstr(prod <= realVar, prodName + "_le_var");
    model.addConstr(prod <= realVarUpperBound * bitVar, prodName + "_le_bit");
    LinExpr lowerBound = LinExpr(realVar);
    lowerBound -= realVarUpperBound * (1.0 - bitVar);
    model.addConstr(prod >= lowerBound, prodName + "_ge");

    product += weight * prod;
  }

  // intVar == sum_k 2^k * b_k
  model.addConstr(intExpansion == intVar, name + "_expansion");
  return product;
}

LinExpr linearizeBinaryTimesVar(CPSolver &model, const CPVar &binVar,
                                           const CPVar &var,
                                           double varUpperBound,
                                           llvm::StringRef prefix) {
  std::string prodName = prefix.str();
  CPVar prod = model.addVar(prodName, REAL, 0.0, varUpperBound);
  model.addConstr(prod <= var, prodName + "_le_var");
  model.addConstr(prod <= varUpperBound * binVar, prodName + "_le_bit");
  LinExpr lowerBound = LinExpr(var);
  lowerBound -= varUpperBound * (1.0 - binVar);
  model.addConstr(prod >= lowerBound, prodName + "_ge");
  return LinExpr(prod);
}

// -------------------------------------------------------------
// GurobiSolver method implementations
// -------------------------------------------------------------

#ifdef DYNAMATIC_ENABLE_HIGHS

/// Returns the column of `var` in the HiGHS model, aborting if the variable was
/// never added to this solver.
static HighsInt getColumn(const std::map<CPVar, int> &variables,
                          const CPVar &var) {
  auto it = variables.find(var);
  if (it == variables.end())
    llvm::report_fatal_error(
        ("Variable " + var.impl->name +
         " was never added to the HiGHS solver! Aborting...")
            .c_str());
  return it->second;
}

HighsSolver::HighsSolver(int timeout, int maxThreads)
    : CPSolver(timeout, HIGHS, maxThreads) {
  model = std::make_unique<Highs>();

  // Suppress outputs to stdout (clashes with the MLIR output file).
  model->setOptionValue("output_flag", false);
  model->setOptionValue("log_to_console", false);

  // Always use the same random seed to make the solution deterministic.
  model->setOptionValue("random_seed", 0);

  if (timeout > 0)
    model->setOptionValue("time_limit", static_cast<double>(timeout));

  if (maxThreads > 0)
    model->setOptionValue("threads", static_cast<HighsInt>(maxThreads));
}

HighsSolver::~HighsSolver() = default;

CPVar HighsSolver::addVar(const std::string &name, VarType type,
                          std::optional<double> lb, std::optional<double> ub) {
  auto var = CPVar(name, type, lb, ub);
  return addVar(var);
}

CPVar HighsSolver::addVar(const CPVar &var) {
  if (names.count(var.impl->name)) {
    llvm::report_fatal_error(
        llvm::Twine("Adding variable with duplicated names is not permitted! "
                    "Aborting... (duplicate: ") +
        var.impl->name + ")");
  }

  double lb = var.impl->lowerBound.value_or(-kHighsInf);
  double ub = var.impl->upperBound.value_or(kHighsInf);
  // Unlike Gurobi, HiGHS has no notion of a binary variable, so instead of
  // relying on the bounds the user supplied (which may be open) we pin boolean
  // variables to [0, 1] ourselves, like Gurobi's GRB_BINARY does.
  if (var.impl->type == BOOLEAN) {
    lb = std::max(lb, 0.0);
    ub = std::min(ub, 1.0);
  }

  // The column is created as a continuous one and only made integral
  // afterwards, since HiGHS cannot specify the type at creation time.
  int col = static_cast<int>(model->getNumCol());
  if (model->addVar(lb, ub) != HighsStatus::kOk)
    llvm::report_fatal_error("Failed to add a variable to the HiGHS model!");

  if (var.impl->type != REAL &&
      model->changeColIntegrality(col, HighsVarType::kInteger) !=
          HighsStatus::kOk) {
    llvm::report_fatal_error(
        "Failed to make a variable integral in the HiGHS model!");
  }

  // Names only show up in the dumped models, so a failure here is not fatal.
  model->passColName(col, var.impl->name);

  variables[var] = col;
  varOrder.push_back(var);
  names.insert(var.impl->name);
  return var;
}

void HighsSolver::addConstr(const TempConstr &constraint,
                            llvm::StringRef constrName) {
  if (!constraint.expr.quadTerms.empty())
    llvm::report_fatal_error(
        "Adding a linear constraint with quadratic terms!");

  const LinExpr &expr = constraint.expr.linexpr;
  std::vector<HighsInt> indices;
  std::vector<double> values;
  indices.reserve(expr.terms.size());
  values.reserve(expr.terms.size());
  for (auto &[var, coeff] : expr.terms) {
    if (coeff == 0.0)
      continue;
    indices.push_back(getColumn(variables, var));
    values.push_back(coeff);
  }

  // Constraints are normalized to "[expr] [pred] 0", so the constant term moves
  // to the right-hand side.
  double rhs = -expr.constant;
  HighsInt numNZ = static_cast<HighsInt>(indices.size());

  HighsStatus st;
  if (constraint.pred == LE) {
    st = model->addRow(-kHighsInf, rhs, numNZ, indices.data(), values.data());
  } else if (constraint.pred == EQ) {
    st = model->addRow(rhs, rhs, numNZ, indices.data(), values.data());
  } else {
    llvm_unreachable("Unknown predicate!");
  }

  if (st != HighsStatus::kOk) {
    llvm::errs() << "Failed to add constraint \"" << constrName
                 << "\" to the HiGHS model!\n";
    llvm::report_fatal_error("Failed to add a constraint to the HiGHS model!");
  }
}

void HighsSolver::addQConstr(const TempConstr &constraint,
                             llvm::StringRef constrName) {
  llvm::report_fatal_error(
      "Quadratic constraints are unavailable for HiGHS! They must be "
      "linearized before being added to this solver.");
}

void HighsSolver::setMaximizeObjective(const LinExpr &expr) {
  // The coefficients are only handed over to the model in `optimize`, since
  // HiGHS stores them per column and all columns must exist by then.
  maxObjective = expr;
}

void HighsSolver::optimize() {
  model->changeObjectiveSense(ObjSense::kMaximize);
  for (auto &[var, coeff] : maxObjective.terms) {
    if (coeff == 0.0)
      continue;
    model->changeColCost(getColumn(variables, var), coeff);
  }
  // NOTE: the objective's constant term is ignored, like in the other backends.

  if (model->run() == HighsStatus::kError) {
    status = ERROR;
    return;
  }

  switch (model->getModelStatus()) {
  case HighsModelStatus::kOptimal:
    status = OPTIMAL;
    break;
  case HighsModelStatus::kTimeLimit:
  case HighsModelStatus::kIterationLimit:
  case HighsModelStatus::kObjectiveBound:
  case HighsModelStatus::kObjectiveTarget:
  case HighsModelStatus::kSolutionLimit:
  case HighsModelStatus::kInterrupt:
    // The solver stopped early; a solution is only usable if one was found.
    status = model->getSolution().value_valid ? NONOPTIMAL : UNKNOWN;
    break;
  case HighsModelStatus::kInfeasible:
    status = INFEASIBLE;
    break;
  case HighsModelStatus::kUnbounded:
  case HighsModelStatus::kUnboundedOrInfeasible:
    status = UNBOUNDED;
    break;
  default:
    status = UNKNOWN;
    break;
  }
}

void HighsSolver::write(llvm::StringRef filePath) const {
  if (model->writeModel(filePath.str()) != HighsStatus::kOk)
    llvm::errs() << "Failed to write the HiGHS model to " << filePath << "!\n";
}

void HighsSolver::writeSol(llvm::StringRef filePath) const {
  if (status != OPTIMAL && status != NONOPTIMAL) {
    llvm::report_fatal_error("Calling writeSol before the model was solved!");
  }

  std::ofstream myfile(filePath.str());
  if (myfile.is_open()) {
    for (const auto &var : varOrder)
      myfile << var.impl->name << " = " << getValue(var) << "\n";
  } else {
    llvm::errs() << "Unable to open file: " << filePath << "!\n";
    llvm::report_fatal_error("Unable to open file!");
  }
}

double HighsSolver::getValue(const CPVar &var) const {
  if (status != OPTIMAL && status != NONOPTIMAL) {
    llvm::errs() << "Solution is not available while retrieving "
                 << var.impl->name << "!\n";
    llvm::report_fatal_error("Cannot retrieve the value of variable!");
  }
  return model->getSolution().col_value[getColumn(variables, var)];
}

double HighsSolver::getObjective() const {
  if (status != OPTIMAL && status != NONOPTIMAL) {
    llvm::report_fatal_error("Cannot retrieve the objective because the "
                             "solution is not available!");
  }
  return model->getObjectiveValue();
}

#endif // DYNAMATIC_ENABLE_HIGHS

} // namespace dynamatic
