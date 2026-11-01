/// ConstraintProgramming.h
/// This header defines a DSL for constraint programming
/// - Users can use overloaded '+', '*', '-', '<', etc. to construct constraints
/// and objectives.
/// - It is designed to be solver agnostic.
/// - The API is designed to look very similar to gurobi's API.
///
/// For example usage of this API, please refer to
/// `dynamatic/unittests/Support/ConstraintProgramming/CPTest.cpp`
#pragma once
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
#include "gurobi_c++.h"
#endif // DYNAMATIC_GUROBI_NOT_INSTALLED

#ifdef DYNAMATIC_ENABLE_HIGHS
/// The HiGHS model class, forward declared so that the heavyweight `Highs.h`
/// does not have to be included by every file that includes this header.
class Highs;
#endif // DYNAMATIC_ENABLE_HIGHS

namespace dynamatic {

enum VarType { REAL, INTEGER, BOOLEAN };
/// Forward declaration
namespace detail {

/// Remark: the user of "ConstraintProgramming.h" should not use anything in
/// detail::.
///
/// This struct contains the actual storage of CPVar
struct CPVarImpl {
  std::string name;
  VarType type;
  // Null value of lowerBound would be -inf
  std::optional<double> lowerBound;
  // Null value of upperBound would be +inf
  std::optional<double> upperBound;
  // Using Var as a key

  CPVarImpl() = default;
  // Explicit constructor:
  // Var newVar = solver.addVariable("newVar", Var::INTEGER, std::nullopt,
  // std::nullopt);
  //
  // Explicit constructor avoids implicit cast from string to Var.
  explicit CPVarImpl(llvm::StringRef name, VarType type,
                     std::optional<double> lowerBound = /* -inf */ std::nullopt,
                     std::optional<double> upperBound = /* +inf */ std::nullopt)
      : name(name), type(type), lowerBound(lowerBound), upperBound(upperBound) {

    if (name.empty() || !std::all_of(name.begin(), name.end(), [](char c) {
          return isalnum(c) || c == '_';
        }))
      llvm::report_fatal_error("CPVar's name must be non empty and only "
                               "contain alphanumeric or _ characters");
  }
};
} // namespace detail

/// A single variable in constraint programming
///
/// Examples:
/// Creating variables:
/// 1. An integer variable without upper and lower bounds
/// auto x = Var("x", Var::INTEGER);
/// 2. An float variable without lower bound and has a upperbound of 1
/// auto y = Var("y", Var::REAL, std::nullopt, 1);
///
/// It is a wrapper class around a pointer
/// reference to the actual storage (CPVarImpl). The rationale behind this is
/// the following:
/// - We want to use DSL-like operation to construct LP constraints and
/// objectives.
/// - Storing the pointer version of CPVar in LinExpr/QuadExpr/CPSolver is very
/// painful, and we cannot overload arithmetic operations easily on them.
/// - Putting the storage directly in CPVar causes a lot of copying, which is a
/// huge overhead when there are many of them.
///
/// For example, the copy assignment
///
/// CPVar a("x", INTEGER);
/// CPVar b("y", INTEGER);
/// b = a; // <--- this one
///
/// Would use the implicitly generated copy constructor to do member wise
/// assignment (this->impl = other.impl;), which only copies the pointer but not
/// the underlying memory.
struct CPVar {
  std::shared_ptr<detail::CPVarImpl> impl;
  CPVar() = default;
  explicit CPVar(llvm::StringRef name, VarType type,
                 std::optional<double> lowerBound = /* -inf */ std::nullopt,
                 std::optional<double> upperBound = /* +inf */ std::nullopt)
      : impl(std::make_shared<detail::CPVarImpl>(name, type, lowerBound,
                                                 upperBound)) {};
  bool operator<(const CPVar &other) const noexcept;
  std::string getName() const;
};

inline std::pair<CPVar, CPVar> makeSortedPair(const CPVar &a, const CPVar &b) {
  return (a < b) ? std::make_pair(a, b) : std::make_pair(b, a);
}

/// An expression in constraint programming.
///
/// Examples:
/// 1. Using operator overloading to construct an expression:
/// auto expr1 = (x + 2 * y);
/// 2. Constructing constraints from 2 expressions
/// auto expr2 = (3 * z);
/// Constraint constr1 = (expr1 <= expr2);
struct LinExpr {

  // The coefficients in the linear expression
  // For instance, for x + 2 * y + 1
  // We have (x, 1) and (y, 2)
  std::map<CPVar, double> terms;
  double constant = 0.0;
  LinExpr() = default;
  LinExpr(const CPVar &v) { terms[v] = 1.0; }
  LinExpr(double value) { constant = value; }
  LinExpr operator-() const {
    LinExpr negated;
    for (auto &[var, coeff] : terms) {
      negated.terms[var] = -coeff;
    }
    negated.constant = -constant;
    return negated;
  }
  std::string writeLp() const;
};

LinExpr operator+(const CPVar &left, double right);
LinExpr operator+(double left, const CPVar &right);
LinExpr operator-(const CPVar &left, double right);
LinExpr operator-(double left, const CPVar &right);
LinExpr operator+(const CPVar &left, const CPVar &right);
LinExpr operator-(const CPVar &left, const CPVar &right);
LinExpr operator+(const LinExpr &left, double right);
LinExpr operator+(double left, const LinExpr &right);
LinExpr operator+(const LinExpr &left, const LinExpr &right);
void operator+=(LinExpr &left, const LinExpr &right);
LinExpr operator-(const LinExpr &left, const LinExpr &right);
void operator-=(LinExpr &left, const LinExpr &right);

LinExpr operator*(double c, const CPVar &v);
LinExpr operator*(const CPVar &v, double c);
LinExpr operator*(double c, const LinExpr &expr);
LinExpr operator*(const LinExpr &expr, double c);

struct QuadExpr {
  LinExpr linexpr;
  std::map<std::pair<CPVar, CPVar>, double> quadTerms;
  QuadExpr() = default;
  QuadExpr(double value) { linexpr = LinExpr(value); }
  QuadExpr(const CPVar &var) { linexpr = LinExpr(var); }
  QuadExpr(const LinExpr &expr) { linexpr = expr; }
};

QuadExpr operator*(const LinExpr &lhs, const LinExpr &rhs);
QuadExpr operator*(double lhs, const QuadExpr &rhs);
QuadExpr operator+(const QuadExpr &lhs, const QuadExpr &rhs);
QuadExpr operator-(const QuadExpr &lhs, const QuadExpr &rhs);
void operator+=(QuadExpr &lhs, const QuadExpr &rhs);
void operator-=(QuadExpr &lhs, const QuadExpr &rhs);

enum Predicate {
  /* <= */ LE,
  /* == */ EQ
};

/// Class for constraints
/// It has the form:
/// [expr] [pred] 0
/// For example:
/// - x + 2 * y - z - 1 <= 0
/// - x + 2 * y - z + 1 <= 0
/// - x + 2 * y - z + 2 == 0
/// The rhs is always 0

// NOTE: this name is borrowed from gurobi (GRBTempConstr)
struct TempConstr {
  // The LHS expression. RHS is omitted because it is always set to zero.
  QuadExpr expr;
  Predicate pred;

  // Write the constraint in the lp format.
  std::string writeLp() const;
};

TempConstr operator<=(const QuadExpr &lhs, const QuadExpr &rhs);
TempConstr operator>=(const QuadExpr &lhs, const QuadExpr &rhs);
TempConstr operator==(const QuadExpr &lhs, const QuadExpr &rhs);

/// Abstract base class for different solvers (e.g., Gurobi, Google's solvers,
/// or Cbc).
///
/// This is overloaded for the Gurobi solver and the Google's OR Tools API.
///
/// The user can use exactly the same API to define variables, add
/// constraints/objectives.
class CPSolver {
public:
  enum Status { OPTIMAL, NONOPTIMAL, INFEASIBLE, UNBOUNDED, UNKNOWN, ERROR };
  Status status = UNKNOWN;

  // LLVM Implementation of rtti functions like dyn_cast<>, isa<> needs these
  // function
  // [START LLVM RTTI prerequisites]
  enum SolverKind {
#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
    GUROBI,
#endif // DYNAMATIC_GUROBI_NOT_INSTALLED
#ifdef DYNAMATIC_ENABLE_CBC
    CBC,
#endif // DYNAMATIC_ENABLE_CBC
#ifdef DYNAMATIC_ENABLE_HIGHS
    HIGHS,
#endif      // DYNAMATIC_ENABLE_HIGHS
    DEFAULT // Dummy option
  };
  SolverKind solverKind;
  SolverKind getKind() const { return solverKind; }
  static inline bool classof(CPSolver const *) { return true; }
  // [END LLVM RTTI prerequisites]

  // Solver timeout in second.
  // If timeout <= 0, then this option is ignored.
  int timeout = -1;

  // Maximum number of threads used in MILP solving
  int maxThreads;
  CPSolver(int timeout, SolverKind k, int maxThreads)
      : solverKind(k), timeout(timeout), maxThreads(maxThreads) {}

  virtual ~CPSolver() = default;
  // Virtual class methods: they provide a unified interface for all available
  // solvers.
  virtual CPVar addVar(const CPVar &var) = 0;
  // Create var, add gurobi var, and then return the created variable
  virtual CPVar addVar(const std::string &name, VarType type,
                       std::optional<double> lb, std::optional<double> ub) = 0;
  virtual void addConstr(const TempConstr &constraint,
                         llvm::StringRef constrName) = 0;
  void addConstr(const TempConstr &constraint) { addConstr(constraint, ""); }
  virtual void addQConstr(const TempConstr &constraint,
                          llvm::StringRef constrName) = 0;
  virtual void setMaximizeObjective(const LinExpr &expr) = 0;
  virtual void optimize() = 0;
  virtual double getValue(const CPVar &var) const = 0;
  virtual double getObjective() const = 0;

  virtual void write(llvm::StringRef filePath) const = 0;

  virtual void writeSol(llvm::StringRef filePath) const = 0;

  std::string symbolizeStatus() {
    if (status == OPTIMAL)
      return "OPTIMAL";
    if (status == NONOPTIMAL)
      return "NONOPTIMAL";
    if (status == INFEASIBLE)
      return "INFEASIBLE";
    if (status == UNBOUNDED)
      return "UNBOUNDED";
    if (status == UNKNOWN)
      return "UNKNOWN";
    return "ERROR";
  }
};

/// Creates the variables and constraints expressing the product of an integer
/// variable with a real variable as a *linear* expression, for the solver
/// backends that cannot handle the quadratic constraint this product would
/// otherwise require.
///
/// The transformation is exact, i.e. it is not a relaxation: it does not
/// change the set of solutions of the variables that already existed.
///
/// `intVar` must take integer values in [0, intVarUpperBound] and `realVar` must
/// take values in [0, realVarUpperBound]. Both bounds have to be enforced
/// elsewhere in the model (or by the variables' own bounds) for the
/// linearization to be valid.
///
/// The integer variable is expanded in binary, i.e. intVar = sum_k 2^k * b_k,
/// and every product b_k * realVar is carried by an auxiliary continuous
/// variable `w` constrained by
///   w <= realVar,                                  w <= realVarUpperBound * b_k,
///   w >= realVar - realVarUpperBound * (1 - b_k),  w >= 0,
/// which forces w = b_k * realVar. The returned expression is the linearization
/// of intVar * realVar itself, and can be used like any other `LinExpr`.
///
/// All created variables are named after `prefix`, which therefore has to be
/// unique within the solver.
LinExpr linearizeIntegerTimesVar(CPSolver &model, const CPVar &intVar,
                                 unsigned intVarUpperBound,
                                 const CPVar &realVar, double realVarUpperBound,
                                 llvm::StringRef prefix);

/// Same as `linearizeIntegerTimesVar`, for the product of a boolean variable
/// with a variable taking values in [0, varUpperBound] (a bound that has to be
/// enforced elsewhere in the model). The boolean variable is already binary, so
/// no binary expansion is needed.
LinExpr linearizeBinaryTimesVar(CPSolver &model, const CPVar &binVar,
                                const CPVar &var, double varUpperBound,
                                llvm::StringRef prefix);

#ifndef DYNAMATIC_GUROBI_NOT_INSTALLED
class GurobiSolver : public CPSolver {
  std::unique_ptr<GRBEnv> env;
  std::map<CPVar, GRBVar> variables;
  std::unique_ptr<GRBModel> model;

  // Track the added names: prevent adding variables with duplicated names
  std::set<std::string> names;

public:
  GurobiSolver(int timeout = -1 /* default = no timeout*/,
               int maxThreads = 4 /* default = maximum 4 threads */);

  CPVar addVar(const CPVar &var) override;

  // Create var, add gurobi var, and then return
  CPVar addVar(const std::string &name, VarType type, std::optional<double> lb,
               std::optional<double> ub) override;
  void addConstr(const TempConstr &constraint,
                 llvm::StringRef constrName) override;
  void addQConstr(const TempConstr &constraint,
                  llvm::StringRef constrName) override;
  void optimize() override;

  void setMaximizeObjective(const LinExpr &expr) override;

  void write(llvm::StringRef filePath) const override {
    model->write(filePath.str());
  }
  void writeSol(llvm::StringRef filePath) const override;

  /// Retrieve the value from the solved MILP
  ///
  /// Example:
  /// auto resultMyVar = solver.getValue(myVar);
  double getValue(const CPVar &var) const override;

  double getObjective() const override;

  // [START LLVM RTTI prerequisites]
  static bool classof(const CPSolver *b) { return b->getKind() == GUROBI; }
  static bool classof(const GurobiSolver *b) { return true; }
  // [END LLVM RTTI prerequisites]
};
#endif // DYNAMATIC_GUROBI_NOT_INSTALLED

#ifdef DYNAMATIC_ENABLE_CBC

namespace detail {
struct CbcSoluParser {
  std::map<std::string, double> results;
  CPSolver::Status status;
  double objectiveValue;
  mlir::LogicalResult parseSolverOutput(llvm::StringRef soluFileName);
};
} // namespace detail

class CbcSolver : public CPSolver {

  std::set<CPVar> variables;
  std::set<std::string> names;

  detail::CbcSoluParser solution;

  // Multiple constraints might have the same name (e.g., the same empty name).
  // So here we use a vector of pairs instead of map
  std::vector<std::pair<std::string, TempConstr>> constraints;
  LinExpr maxObjective;

public:
  CbcSolver(int timeout = -1 /* default = no timeout */,
            int maxThreads = -1 /* note: currently this option has no effect */)
      : CPSolver(timeout, CBC, maxThreads) {}

  CPVar addVar(const CPVar &var) override;
  CPVar addVar(const std::string &name, VarType type, std::optional<double> lb,
               std::optional<double> ub) override;
  void addConstr(const TempConstr &constraint,
                 llvm::StringRef constrName) override;

  void addQConstr(const TempConstr &constraint, llvm::StringRef name) override {
    llvm::report_fatal_error(
        "Quadratic constraints is currently unavailable for CBC!");
  }
  void setMaximizeObjective(const LinExpr &expr) override;
  void optimize() override;
  void write(llvm::StringRef filePath) const override {
    this->writeLp(filePath);
  }

  void writeSol(llvm::StringRef filePath) const override;

  double getValue(const CPVar &var) const override;

  double getObjective() const override;

  void writeLp(llvm::StringRef) const;

  // [START LLVM RTTI prerequisites]
  static bool classof(const CbcSolver *b) { return true; }
  static bool classof(const CPSolver *b) { return b->getKind() == CBC; }
  // [END LLVM RTTI prerequisites]
};

#endif // DYNAMATIC_ENABLE_CBC

#ifdef DYNAMATIC_ENABLE_HIGHS

/// Solver backend built on top of HiGHS (https://github.com/ERGO-Code/HiGHS),
/// an open-source (MIT licensed) MILP solver. Contrarily to the CBC backend,
/// HiGHS is linked into the binary and solves the model in-process, so no
/// external executable is required at runtime.
///
/// Note that HiGHS only supports linear constraints, hence the quadratic
/// constraints used by some of Dynamatic's MILPs have to be linearized before
/// they reach this backend (see
/// `BufferPlacementMILP::addChannelThroughputConstraintsForIntegerLatencyChannel`).
class HighsSolver : public CPSolver {

  /// The HiGHS model doing all the work.
  std::unique_ptr<Highs> model;

  /// Column index in the HiGHS model of every constraint programming variable.
  std::map<CPVar, int> variables;

  /// Variables in the order they were added, used to dump solutions.
  std::vector<CPVar> varOrder;

  /// Track the added names: prevent adding variables with duplicated names
  std::set<std::string> names;

  /// The objective to maximize. It is only handed to the model right before
  /// optimization, since HiGHS stores objective coefficients per column and
  /// every column must exist by then.
  LinExpr maxObjective;

public:
  HighsSolver(int timeout = -1 /* default = no timeout*/,
              int maxThreads = 4 /* default = maximum 4 threads */);

  /// Defined out of line, where `Highs` is a complete type (it is only forward
  /// declared in this header).
  ~HighsSolver() override;

  CPVar addVar(const CPVar &var) override;

  // Create var, add highs var, and then return
  CPVar addVar(const std::string &name, VarType type, std::optional<double> lb,
               std::optional<double> ub) override;
  void addConstr(const TempConstr &constraint,
                 llvm::StringRef constrName) override;

  void addQConstr(const TempConstr &constraint, llvm::StringRef constrName) override;

  void optimize() override;

  void setMaximizeObjective(const LinExpr &expr) override;

  void write(llvm::StringRef filePath) const override;
  void writeSol(llvm::StringRef filePath) const override;

  /// Retrieve the value from the solved MILP
  ///
  /// Example:
  /// auto resultMyVar = solver.getValue(myVar);
  double getValue(const CPVar &var) const override;

  double getObjective() const override;

  // [START LLVM RTTI prerequisites]
  static bool classof(const CPSolver *b) { return b->getKind() == HIGHS; }
  static bool classof(const HighsSolver *b) { return true; }
  // [END LLVM RTTI prerequisites]
};

#endif // DYNAMATIC_ENABLE_HIGHS

} // namespace dynamatic
