use std::fmt::{self, Display};
use Inst::{Flow, Linear};

pub type Ident = String;
pub type Label = String;
pub type ArgStackOffset = usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum JumpMode {
    Unconditional,
    Zero,
    NonZero,
}

impl JumpMode {
    pub fn rev(self) -> Self {
        match self {
            JumpMode::Unconditional => JumpMode::Unconditional,
            JumpMode::Zero => JumpMode::NonZero,
            JumpMode::NonZero => JumpMode::Zero,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    Plus,
    Minus,
    Mul,
    Div,
    Mod,
    Eq,
    Gt,
    GtEq,
    Lt,
    LtEq,
    NotEq,
    And,
}

impl Op {
    pub fn eval(&self, lhs: i32, rhs: i32) -> i32 {
        match self {
            Op::Plus => lhs + rhs,
            Op::Minus => lhs - rhs,
            Op::Mul => lhs * rhs,
            Op::Eq => (lhs == rhs) as i32,
            Op::Gt => (lhs > rhs) as i32,
            Op::Lt => (lhs < rhs) as i32,
            Op::Div => lhs / rhs,
            Op::Mod => lhs % rhs,
            Op::GtEq => (lhs >= rhs) as i32,
            Op::LtEq => (lhs <= rhs) as i32,
            Op::NotEq => (lhs != rhs) as i32,
            Op::And => ((lhs != 0) && (rhs != 0)) as i32,
        }
    }
}

impl Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Op::Plus => write!(f, "+"),
            Op::Minus => write!(f, "-"),
            Op::Mul => write!(f, "*"),
            Op::Div => write!(f, "/"),
            Op::Mod => write!(f, "%"),
            Op::Eq => write!(f, "=="),
            Op::Gt => write!(f, ">"),
            Op::Lt => write!(f, "<"),
            Op::GtEq => write!(f, ">="),
            Op::LtEq => write!(f, "<="),
            Op::NotEq => write!(f, "!="),
            Op::And => write!(f, "&&"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Sym {
    Arg(u16),
    Loc(u16),
    Glb(Ident),
    Acc(u16),
}

impl Display for Sym {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Sym::Arg(v) => write!(f, "arg[{v}]"),
            Sym::Loc(v) => write!(f, "loc[{v}]"),
            Sym::Glb(id) => write!(f, "{id}"),
            Sym::Acc(v) => write!(f, "acc[{v}]"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Pat {
    Tag(Ident, usize),
    Array,
    Sexp,
    String,
    UnBoxed,
    Closure,
    Boxed,
}

impl Display for Pat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Pat::Tag(t, args) => write!(f, "Tag ({t}, {args})"),
            Pat::Array => write!(f, "Array"),
            Pat::Sexp => write!(f, "Sexp"),
            Pat::String => write!(f, "String"),
            Pat::UnBoxed => write!(f, "UnBoxed"),
            Pat::Closure => write!(f, "Closure"),
            Pat::Boxed => write!(f, "Boxed"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinInst {
    Const(i32),
    String(String),
    Array(usize),
    Elem,
    Store(Sym),
    Load(Sym),
    BinOp(Op),
    Pat(Pat),
    SExp(Ident, usize),
    Closure(Ident, usize),
    LDA(Sym),
    Dup,
    Drop,
}

impl Display for LinInst {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LinInst::Const(v) => write!(f, "CONST {v}"),
            LinInst::Elem => write!(f, "ELEM"),
            LinInst::Store(sym) => write!(f, "ST {sym}"),
            LinInst::Load(sym) => write!(f, "LD {sym}"),
            LinInst::BinOp(op) => write!(f, "BINOP {op}"),
            LinInst::Pat(pat) => write!(f, "PATT {pat}"),
            LinInst::SExp(t, args) => write!(f, "SEXP {t}, {args}"),
            LinInst::Dup => write!(f, "DUP"),
            LinInst::Drop => write!(f, "DROP"),
            LinInst::LDA(sym) => write!(f, "LDA {sym}"),
            LinInst::String(v) => write!(f, "STRING {v}"),
            LinInst::Array(n) => write!(f, "ARRAY {n}"),
            LinInst::Closure(name, captures) => write!(f, "CLOSURE {name}, {captures}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Begin {
    pub name: Ident,
    pub args: u32,
    pub locs: u32,
    pub clos: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LabelMode {
    DropBarrier,
    RetrieveStack,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FlowInst {
    Jmp(JumpMode, Ident),
    Label(Ident, LabelMode),
    // call command should be in flow graph, because it can cause
    // arbitrary change of global variables. And after unfolding it can
    // have control flow instruction which weird to extract from lin block
    Call(Ident, usize),
    // STI command should be in flow graph, because analysis of variable's values
    // at the end of the linear block
    STI,
    STA,
    CallC(usize),
    Begin(Begin),
    End,
}

impl FlowInst {
    pub fn conditional_jmp(&self) -> bool {
        matches!(
            self,
            FlowInst::Jmp(JumpMode::NonZero, _) | FlowInst::Jmp(JumpMode::Zero, _)
        )
    }
}

impl Display for LabelMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LabelMode::DropBarrier => write!(f, "1"),
            LabelMode::RetrieveStack => write!(f, "0"),
        }
    }
}

impl Display for FlowInst {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FlowInst::Jmp(mode, label) => match mode {
                JumpMode::Unconditional => write!(f, "JMP {label}"),
                JumpMode::Zero => write!(f, "CJMP z, {label}"),
                JumpMode::NonZero => write!(f, "CJMP nz, {label}"),
            },
            FlowInst::Label(label, mode) => write!(f, "LABEL {label}, {mode}"),
            FlowInst::Call(name, args) => write!(f, "CALL {name}, {args}"),
            FlowInst::CallC(args) => write!(f, "CALLC {args}"),
            FlowInst::STI => write!(f, "STI"),
            FlowInst::STA => write!(f, "STA"),
            FlowInst::Begin(begin) => write!(
                f,
                "BEGIN {}, {}, {}, {}",
                begin.name, begin.args, begin.locs, begin.clos
            ),
            FlowInst::End => write!(f, "END"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Public {
    Val(Ident, Ident),
    Var(Ident, Ident),
    Fun(Ident, Ident, u64),
}

impl fmt::Display for Public {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Public::Val(unit, ident) => write!(f, "PUBLIC Val ({unit}, {ident})"),
            Public::Var(unit, ident) => write!(f, "PUBLIC Var ({unit}, {ident})"),
            Public::Fun(unit, ident, args) => write!(f, "PUBLIC Fun ({unit}, {ident}, {args})"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decl {
    Global(Ident),
    Public(Public),
}

impl fmt::Display for Decl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Decl::Global(var) => write!(f, "GLOBAL {var}"),
            Decl::Public(public) => public.fmt(f),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Ctx {
    pub free_label: u32,
}

impl Ctx {
    pub fn fresh_label(&mut self) -> String {
        let label = format!("Lfresh_{}", self.free_label);
        self.free_label += 1;
        label
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DataGraphOptimFlags {
    pub elim_dead_code: bool,
    pub elim_stores: bool,
    pub const_prop: bool,
    pub tag_eval: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct FlowOptimFlags {
    pub elim_dead_code: bool,
    pub jump_on_const: bool,
    pub merge_blocks: bool,
    pub liveliness_analysis: bool,
    pub tail_call: bool,
    pub data_flags: DataGraphOptimFlags,
    pub passes: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Inst {
    Linear(LinInst),
    Flow(FlowInst),
    Decl(Decl),
}

impl Inst {
    fn parse_sym<'a, I: Iterator<Item = &'a str>>(tokens: &mut I) -> Option<Sym> {
        match tokens.next()? {
            "arg" => {
                let id = tokens.next()?.parse().ok()?;
                Some(Sym::Arg(id))
            }
            "loc" => {
                let id = tokens.next()?.parse().ok()?;
                Some(Sym::Loc(id))
            }
            "acc" => {
                let id = tokens.next()?.parse().ok()?;
                Some(Sym::Acc(id))
            }
            other => Some(Sym::Glb(other.to_string())),
        }
    }

    pub fn parse(code: &str) -> Option<Inst> {
        let mut tokens = code
            .split(|ch| match ch {
                ',' | '(' | ')' | '[' | ']' => true,
                _ if ch.is_whitespace() => true,
                _ => false,
            })
            .filter(|token| !token.is_empty());

        match tokens.next()? {
            "BEGIN" => {
                let name = tokens.next()?.to_string();
                let args = tokens.next()?.parse().ok()?;
                let locs = tokens.next()?.parse().ok()?;
                let clos = tokens.next()?.parse().ok()?;
                Some(Flow(FlowInst::Begin(Begin {
                    name,
                    args,
                    locs,
                    clos,
                })))
            }
            "END" => Some(Flow(FlowInst::End)),
            "LD" => {
                let sym = Inst::parse_sym(&mut tokens)?;
                Some(Linear(LinInst::Load(sym)))
            }
            "ST" => {
                let sym = Inst::parse_sym(&mut tokens)?;
                Some(Linear(LinInst::Store(sym)))
            }
            "LDA" => {
                let sym = Inst::parse_sym(&mut tokens)?;
                Some(Linear(LinInst::LDA(sym)))
            }
            "STI" => Some(Flow(FlowInst::STI)),
            "STA" => Some(Flow(FlowInst::STA)),
            "DUP" => Some(Linear(LinInst::Dup)),
            "DROP" => Some(Linear(LinInst::Drop)),
            "LABEL" => {
                let label = tokens.next()?.to_string();
                // LabelMode from original SM code is useless
                Some(Flow(FlowInst::Label(label, LabelMode::DropBarrier)))
            }
            "CONST" => {
                let value = tokens.next()?.parse().ok()?;
                Some(Linear(LinInst::Const(value)))
            }
            "STRING" => {
                // FIXME: parse like normal human being
                let value = code.strip_prefix("STRING ")?.to_string();
                Some(Linear(LinInst::String(value)))
            }
            "ARRAY" => {
                let n = tokens.next()?.parse().ok()?;
                Some(Linear(LinInst::Array(n)))
            }
            "BINOP" => match tokens.next() {
                Some("+") => Some(Linear(LinInst::BinOp(Op::Plus))),
                Some("-") => Some(Linear(LinInst::BinOp(Op::Minus))),
                Some("*") => Some(Linear(LinInst::BinOp(Op::Mul))),
                Some("/") => Some(Linear(LinInst::BinOp(Op::Div))),
                Some("%") => Some(Linear(LinInst::BinOp(Op::Mod))),
                Some("==") => Some(Linear(LinInst::BinOp(Op::Eq))),
                Some(">") => Some(Linear(LinInst::BinOp(Op::Gt))),
                Some(">=") => Some(Linear(LinInst::BinOp(Op::GtEq))),
                Some("<") => Some(Linear(LinInst::BinOp(Op::Lt))),
                Some("<=") => Some(Linear(LinInst::BinOp(Op::LtEq))),
                Some("!=") => Some(Linear(LinInst::BinOp(Op::NotEq))),
                Some("&&") => Some(Linear(LinInst::BinOp(Op::And))),
                x => panic!("unknown binary op: {x:?}"),
            },
            "JMP" => {
                let label = tokens.next()?.to_string();
                Some(Flow(FlowInst::Jmp(JumpMode::Unconditional, label)))
            }
            "CJMP" => {
                let mode = match tokens.next() {
                    Some("z") => JumpMode::Zero,
                    Some("nz") => JumpMode::NonZero,
                    _ => panic!("unknown jump command"),
                };

                let label = tokens.next()?.to_string();
                Some(Flow(FlowInst::Jmp(mode, label)))
            }
            "ELEM" => Some(Linear(LinInst::Elem)),
            "PATT" => {
                let pat = match tokens.next()? {
                    "Tag" => {
                        let tag = tokens.next()?.strip_prefix('\"')?.strip_suffix('\"')?;
                        let num = tokens.next()?.parse().ok()?;
                        Some(Pat::Tag(tag.to_string(), num))
                    }
                    "Array" => Some(Pat::Array),
                    "Sexp" => Some(Pat::Sexp),
                    "Strin" => Some(Pat::String),
                    "UnBox" => Some(Pat::UnBoxed),
                    "Closu" => Some(Pat::Closure),
                    "Boxed" => Some(Pat::Boxed),
                    _ => None,
                }?;
                Some(Linear(LinInst::Pat(pat)))
            }
            "SEXP" => {
                let tag = tokens.next()?.strip_prefix('\"')?.strip_suffix('\"')?;
                let num = tokens.next()?.parse().ok()?;
                Some(Linear(LinInst::SExp(tag.to_string(), num)))
            }
            "CALL" => {
                let name = tokens.next()?.to_string();
                let num = tokens.next()?.parse().ok()?;
                Some(Flow(FlowInst::Call(name, num)))
            }
            "CLOSURE" => {
                let name = tokens.next()?.to_string();
                let captures = tokens.next()?.parse().ok()?;
                Some(Linear(LinInst::Closure(name, captures)))
            }
            "CALLC" => {
                let args = tokens.next()?.parse().ok()?;
                Some(Flow(FlowInst::CallC(args)))
            }
            "PUBLIC" => match tokens.next()? {
                "Val" => {
                    let unit = tokens.next()?.to_string();
                    let name = tokens.next()?.to_string();
                    Some(Inst::Decl(Decl::Public(Public::Val(unit, name))))
                }
                "Var" => {
                    let unit = tokens.next()?.to_string();
                    let name = tokens.next()?.to_string();
                    Some(Inst::Decl(Decl::Public(Public::Var(unit, name))))
                }
                "Fun" => {
                    let unit = tokens.next()?.to_string();
                    let name = tokens.next()?.to_string();
                    let args = tokens.next()?.parse().ok()?;
                    Some(Inst::Decl(Decl::Public(Public::Fun(unit, name, args))))
                }
                _ => panic!("unknown public"),
            },
            "GLOBAL" => {
                let global = tokens.next()?.to_string();
                Some(Inst::Decl(Decl::Global(global)))
            }
            other => panic!("unknown command: {other}"),
        }
    }
}

impl Display for Inst {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Inst::Linear(inst) => inst.fmt(f),
            Inst::Flow(inst) => inst.fmt(f),
            Inst::Decl(decl) => decl.fmt(f),
        }
    }
}
