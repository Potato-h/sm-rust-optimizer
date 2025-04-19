use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{self, Display, Write};
use std::fs::{self, File};
use std::io::read_to_string;
use std::mem;
use std::path::PathBuf;

use clap::Parser;
use either::Either::{self, Left, Right};
use itertools::Itertools;
use petgraph::csr::IndexType;
use petgraph::visit::{EdgeRef, IntoEdgeReferences, IntoNodeReferences};
use petgraph::{algo, prelude::*, EdgeType};
use Inst::*;

type Ident = String;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum JumpMode {
    Unconditional,
    Zero,
    NonZero,
}

impl JumpMode {
    fn rev(self) -> Self {
        match self {
            JumpMode::Unconditional => JumpMode::Unconditional,
            JumpMode::Zero => JumpMode::NonZero,
            JumpMode::NonZero => JumpMode::Zero,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Op {
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
    fn eval(&self, lhs: i32, rhs: i32) -> i32 {
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
enum Sym {
    Arg(u16),
    Loc(u16),
    Glb(Ident),
}

impl Display for Sym {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Sym::Arg(v) => write!(f, "arg[{v}]"),
            Sym::Loc(v) => write!(f, "loc[{v}]"),
            Sym::Glb(id) => write!(f, "{id}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum LinInst {
    Const(i32),
    String(String),
    Array(usize),
    Elem,
    Store(Sym),
    Load(Sym),
    BinOp(Op),
    Tag(Ident, usize),
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
            LinInst::Tag(t, args) => write!(f, "PATT Tag ({t}, {args})"),
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
struct Begin {
    name: Ident,
    args: u32,
    locs: u32,
    clos: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum FlowInst {
    Jmp(JumpMode, Ident),
    Label(Ident),
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
    fn conditional_jmp(&self) -> bool {
        matches!(
            self,
            FlowInst::Jmp(JumpMode::NonZero, _) | FlowInst::Jmp(JumpMode::Zero, _)
        )
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
            FlowInst::Label(label) => write!(f, "LABEL {label}, 1"),
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
enum Public {
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
enum Decl {
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
struct Ctx {
    free_label: u32,
}

impl Ctx {
    fn fresh_label(&mut self) -> String {
        let label = format!("Lfresh_{}", self.free_label);
        self.free_label += 1;
        label
    }
}

#[derive(Debug, Clone, Copy)]
struct DataGraphOptimFlags {
    elim_dead_code: bool,
    elim_stores: bool,
    const_prop: bool,
    tag_eval: bool,
}

#[derive(Debug, Clone, Copy)]
struct FlowOptimFlags {
    elim_dead_code: bool,
    jump_on_const: bool,
    merge_blocks: bool,
    data_flags: DataGraphOptimFlags,
    passes: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Inst {
    LinInst(LinInst),
    FlowInst(FlowInst),
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
            other => Some(Sym::Glb(other.to_string())),
        }
    }

    fn parse(code: &str) -> Option<Inst> {
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
                Some(FlowInst(FlowInst::Begin(Begin {
                    name,
                    args,
                    locs,
                    clos,
                })))
            }
            "END" => Some(FlowInst(FlowInst::End)),
            "LD" => {
                let sym = Inst::parse_sym(&mut tokens)?;
                Some(LinInst(LinInst::Load(sym)))
            }
            "ST" => {
                let sym = Inst::parse_sym(&mut tokens)?;
                Some(LinInst(LinInst::Store(sym)))
            }
            "LDA" => {
                let sym = Inst::parse_sym(&mut tokens)?;
                Some(LinInst(LinInst::LDA(sym)))
            }
            "STI" => Some(FlowInst(FlowInst::STI)),
            "STA" => Some(FlowInst(FlowInst::STA)),
            "DUP" => Some(LinInst(LinInst::Dup)),
            "DROP" => Some(LinInst(LinInst::Drop)),
            "LABEL" => {
                let label = tokens.next()?.to_string();
                Some(FlowInst(FlowInst::Label(label)))
            }
            "CONST" => {
                let value = tokens.next()?.parse().ok()?;
                Some(LinInst(LinInst::Const(value)))
            }
            "STRING" => {
                // FIXME: parse like normal human being
                let value = code.strip_prefix("STRING ")?.to_string();
                Some(LinInst(LinInst::String(value)))
            }
            "ARRAY" => {
                let n = tokens.next()?.parse().ok()?;
                Some(LinInst(LinInst::Array(n)))
            }
            "BINOP" => match tokens.next() {
                Some("+") => Some(LinInst(LinInst::BinOp(Op::Plus))),
                Some("-") => Some(LinInst(LinInst::BinOp(Op::Minus))),
                Some("*") => Some(LinInst(LinInst::BinOp(Op::Mul))),
                Some("/") => Some(LinInst(LinInst::BinOp(Op::Div))),
                Some("%") => Some(LinInst(LinInst::BinOp(Op::Mod))),
                Some("==") => Some(LinInst(LinInst::BinOp(Op::Eq))),
                Some(">") => Some(LinInst(LinInst::BinOp(Op::Gt))),
                Some(">=") => Some(LinInst(LinInst::BinOp(Op::GtEq))),
                Some("<") => Some(LinInst(LinInst::BinOp(Op::Lt))),
                Some("<=") => Some(LinInst(LinInst::BinOp(Op::LtEq))),
                Some("!=") => Some(LinInst(LinInst::BinOp(Op::NotEq))),
                Some("&&") => Some(LinInst(LinInst::BinOp(Op::And))),
                x => panic!("unknown binary op: {x:?}"),
            },
            "JMP" => {
                let label = tokens.next()?.to_string();
                Some(FlowInst(FlowInst::Jmp(JumpMode::Unconditional, label)))
            }
            "CJMP" => {
                let mode = match tokens.next() {
                    Some("z") => JumpMode::Zero,
                    Some("nz") => JumpMode::NonZero,
                    _ => panic!("unknown jump command"),
                };

                let label = tokens.next()?.to_string();
                Some(FlowInst(FlowInst::Jmp(mode, label)))
            }
            "ELEM" => Some(LinInst(LinInst::Elem)),
            "PATT" => {
                let _ = tokens.next();
                let tag = tokens.next()?.strip_prefix('\"')?.strip_suffix('\"')?;
                let num = tokens.next()?.parse().ok()?;
                Some(LinInst(LinInst::Tag(tag.to_string(), num)))
            }
            "SEXP" => {
                let tag = tokens.next()?.strip_prefix('\"')?.strip_suffix('\"')?;
                let num = tokens.next()?.parse().ok()?;
                Some(LinInst(LinInst::SExp(tag.to_string(), num)))
            }
            "CALL" => {
                let name = tokens.next()?.to_string();
                let num = tokens.next()?.parse().ok()?;
                Some(FlowInst(FlowInst::Call(name, num)))
            }
            "CLOSURE" => {
                let name = tokens.next()?.to_string();
                let captures = tokens.next()?.parse().ok()?;
                Some(LinInst(LinInst::Closure(name, captures)))
            }
            "CALLC" => {
                let args = tokens.next()?.parse().ok()?;
                Some(FlowInst(FlowInst::CallC(args)))
            }
            "PUBLIC" => match tokens.next()? {
                "Val" => {
                    let unit = tokens.next()?.to_string();
                    let name = tokens.next()?.to_string();
                    Some(Decl(Decl::Public(Public::Val(unit, name))))
                }
                "Var" => {
                    let unit = tokens.next()?.to_string();
                    let name = tokens.next()?.to_string();
                    Some(Decl(Decl::Public(Public::Var(unit, name))))
                }
                "Fun" => {
                    let unit = tokens.next()?.to_string();
                    let name = tokens.next()?.to_string();
                    let args = tokens.next()?.parse().ok()?;
                    Some(Decl(Decl::Public(Public::Fun(unit, name, args))))
                }
                _ => panic!("unknown public"),
            },
            "GLOBAL" => {
                let global = tokens.next()?.to_string();
                Some(Decl(Decl::Global(global)))
            }
            other => panic!("unknown command: {other}"),
        }
    }
}

impl Display for Inst {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Inst::LinInst(inst) => inst.fmt(f),
            Inst::FlowInst(inst) => inst.fmt(f),
            Inst::Decl(decl) => decl.fmt(f),
        }
    }
}

type ArgStackOffset = usize;

#[derive(Debug, Clone)]
struct DataGraph {
    start_label: String,
    dag: StableGraph<DataVertex, ArgStackOffset>,
    inputs: Vec<NodeIndex>,
    outputs: VirtualStack,
    symbolics: BTreeMap<Sym, NodeIndex>,
    jump_decided_by: Option<NodeIndex>,
}

impl DataGraph {
    fn outputs_nodes(&self) -> &[NodeIndex] {
        &self.outputs.stack
    }

    fn info_label(&self) -> String {
        format!(
            "{}\ninputs: {}\noutputs: {}",
            self.start_label,
            self.inputs.len(),
            self.outputs_nodes().len()
        )
    }

    fn args_count(&self) -> u16 {
        self.symbolics
            .keys()
            .filter_map(|x| match x {
                Sym::Arg(x) => Some(x + 1),
                _ => None,
            })
            .max()
            .unwrap_or(0)
    }

    fn locs_count(&self) -> u16 {
        self.symbolics
            .keys()
            .filter_map(|x| match x {
                Sym::Loc(x) => Some(x + 1),
                _ => None,
            })
            .max()
            .unwrap_or(0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JumpCondition {
    Unconditional,
    Zero,
    NonZero,
}

impl JumpCondition {
    fn will_jump(&self, value: i32) -> bool {
        match self {
            JumpCondition::Unconditional => true,
            JumpCondition::Zero if value == 0 => true,
            JumpCondition::NonZero if value != 0 => true,
            _ => false,
        }
    }

    fn is_conditional(&self) -> bool {
        matches!(self, JumpCondition::NonZero | JumpCondition::Zero)
    }
}

impl From<JumpMode> for JumpCondition {
    fn from(value: JumpMode) -> Self {
        match value {
            JumpMode::Unconditional => JumpCondition::Unconditional,
            JumpMode::Zero => JumpCondition::Zero,
            JumpMode::NonZero => JumpCondition::NonZero,
        }
    }
}

#[derive(Debug, Clone)]
enum DataVertex {
    Symbolic(Sym),
    StackVar(ArgStackOffset),
    OpResult(LinInst),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct VirtualStack {
    tail_from: usize,
    stack: Vec<NodeIndex>,
}

impl VirtualStack {
    fn new() -> Self {
        VirtualStack {
            tail_from: 0,
            stack: Vec::new(),
        }
    }

    fn push(&mut self, v: NodeIndex) {
        self.stack.push(v);
    }

    fn pop(&mut self) -> Either<DataVertex, NodeIndex> {
        match self.stack.pop() {
            Some(v) => Right(v),
            None => {
                let tail = self.tail_from;
                self.tail_from += 1;
                Left(DataVertex::StackVar(tail))
            }
        }
    }

    fn peek(&self) -> Either<DataVertex, NodeIndex> {
        match self.stack.last() {
            Some(&v) => Right(v),
            None => {
                let tail = self.tail_from;
                Left(DataVertex::StackVar(tail))
            }
        }
    }
}

#[derive(Debug, Clone)]
struct CallVertex {
    label: String,
    name: String,
    args: usize,
}

#[derive(Debug, Clone)]
struct STIVertex {
    label: String,
}

#[derive(Debug, Clone)]
struct STAVertex {
    label: String,
}

#[derive(Debug, Clone)]
struct CallCVertex {
    label: String,
    args: usize,
}

#[derive(Debug, Clone)]
enum FlowVertex {
    LinearBlock(DataGraph),
    Call(CallVertex),
    STI(STIVertex),
    STA(STAVertex),
    CallC(CallCVertex),
}

impl FlowVertex {
    fn is_block(&self) -> bool {
        matches!(self, Self::LinearBlock(_))
    }

    fn block(&self) -> Option<&DataGraph> {
        match self {
            FlowVertex::LinearBlock(graph) => Some(graph),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
struct FlowGraph {
    graph: StableGraph<FlowVertex, JumpCondition>,
    input: NodeIndex,
    outputs: Vec<NodeIndex>,
}

fn inst_vertex(inst: &LinInst) -> DataVertex {
    DataVertex::OpResult(inst.clone())
}

fn analyze_lin_block(start_label: String, code: Vec<LinInst>, has_cjmp: bool) -> DataGraph {
    let mut dag = StableGraph::new();
    let mut stack = VirtualStack::new();
    let mut symbolics: BTreeMap<Sym, NodeIndex> = BTreeMap::new();

    fn n_arity_inst(
        dag: &mut StableGraph<DataVertex, usize>,
        stack: &mut VirtualStack,
        n: usize,
        inst: LinInst,
    ) {
        let node = dag.add_node(inst_vertex(&inst));

        for i in 0..n {
            let arg = stack.pop().right_or_else(|var| dag.add_node(var));
            dag.add_edge(arg, node, i);
        }

        stack.push(node);
    }

    eprintln!("code: {code:?}");

    for inst in code {
        eprintln!("stack: {stack:?}");

        // Assume, that all of the instructions consume stack values, aside from ST
        match inst {
            LinInst::Const(_) => n_arity_inst(&mut dag, &mut stack, 0, inst),
            LinInst::String(_) => n_arity_inst(&mut dag, &mut stack, 0, inst),
            LinInst::Elem => n_arity_inst(&mut dag, &mut stack, 2, inst),
            LinInst::Store(ref sym) => {
                let node = dag.add_node(inst_vertex(&inst));
                let from = match stack.peek() {
                    Left(v) => {
                        let node = dag.add_node(v);
                        stack.pop(); // move lazy input stack variable
                        stack.push(node); // reify lazy input stack variable
                        node
                    }
                    Right(node) => node,
                };
                dag.add_edge(from, node, 0);
                symbolics.insert(sym.clone(), node);
            }
            LinInst::Load(ref sym) => {
                if !symbolics.contains_key(&sym) {
                    let node = dag.add_node(DataVertex::Symbolic(sym.clone()));
                    symbolics.insert(sym.clone(), node);
                }

                stack.push(symbolics[&sym]);
            }
            LinInst::LDA(_) => n_arity_inst(&mut dag, &mut stack, 0, inst),
            LinInst::BinOp(_) => n_arity_inst(&mut dag, &mut stack, 2, inst),
            LinInst::Tag(_, _) => n_arity_inst(&mut dag, &mut stack, 1, inst),
            LinInst::SExp(_, n) => n_arity_inst(&mut dag, &mut stack, n, inst),
            LinInst::Closure(_, n) => n_arity_inst(&mut dag, &mut stack, n, inst),
            LinInst::Array(n) => n_arity_inst(&mut dag, &mut stack, n, inst),
            LinInst::Dup => {
                match stack.peek() {
                    Left(stack_input_var) => {
                        let index = dag.add_node(stack_input_var);
                        stack.pop(); // move lazy input to next variable
                        stack.push(index); // reify input stack variable
                        stack.push(index); // actually perform DUP operation
                    }
                    Right(index) => {
                        stack.push(index);
                    }
                }
            }
            LinInst::Drop => {
                // TODO: actually, is this right? we can do drop at the start of linear block,
                // but silent `pop` will just ignore dependency from blocks into that block
                stack.pop().right_or_else(|var| dag.add_node(var));
            }
        }
    }

    let jump_decided_by = if has_cjmp {
        Some(stack.pop().right_or_else(|var| dag.add_node(var)))
    } else {
        None
    };

    let inputs = dag
        .node_references()
        .filter_map(|(i, typ)| match typ {
            DataVertex::StackVar(_) => Some(i),
            DataVertex::Symbolic(_) => Some(i),
            DataVertex::OpResult(_) => None,
        })
        .collect();

    assert!(
        !algo::is_cyclic_directed(&dag),
        "found cycle in data flow graph"
    );

    DataGraph {
        start_label,
        dag,
        inputs,
        outputs: stack,
        symbolics,
        jump_decided_by,
    }
}

/// Insert graph `extend` into graph `source` and return mapping of old `extend` indexes
/// into indexes into updated `source`.
fn add_graph<N, E, Ty, Ix>(
    source: &mut StableGraph<N, E, Ty, Ix>,
    extend: &StableGraph<N, E, Ty, Ix>,
) -> BTreeMap<NodeIndex<Ix>, NodeIndex<Ix>>
where
    Ix: IndexType,
    Ty: EdgeType,
    N: Clone,
    E: Clone,
{
    let mut extend_to_source = BTreeMap::new();

    for (id, w) in extend.node_references() {
        let new_id = source.add_node(w.clone());
        extend_to_source.insert(id, new_id);
    }

    for edge in extend.edge_references() {
        source.add_edge(
            extend_to_source[&edge.source()],
            extend_to_source[&edge.target()],
            edge.weight().clone(),
        );
    }

    extend_to_source
}

impl DataGraph {
    fn extend(&mut self, ext: &DataGraph) {
        let ext_to_source = add_graph(&mut self.dag, &ext.dag);

        let ext_stack_vars_in_source = ext.inputs.iter().filter_map(|&node| match ext.dag[node] {
            DataVertex::StackVar(offset) => Some((ext_to_source[&node], offset)),
            _ => None,
        });

        let mut ext_deleted_inputs = BTreeMap::new();

        for (input, _) in ext_stack_vars_in_source.sorted_by_key(|(_, offset)| *offset) {
            let output = self.outputs.pop().right_or_else(|var| {
                let new_stack_arg = self.dag.add_node(var);
                self.inputs.push(new_stack_arg);
                new_stack_arg
            });

            let input_outgoings = self
                .dag
                .edges_directed(input, Direction::Outgoing)
                .map(|edge| (edge.target(), *edge.weight()))
                .collect_vec();

            for (to, w) in input_outgoings {
                self.dag.add_edge(output, to, w);
            }

            ext_deleted_inputs.insert(input, output);
            self.dag.remove_node(input);
        }

        // Merge results of 2 stacks. Now self contains actual count of
        // virtual nodes, but ext contains actual output nodes that will be on stack
        // after block execution
        self.outputs
            .stack
            .extend(ext.outputs.stack.iter().map(|node| {
                *ext_deleted_inputs
                    .get(&ext_to_source[node])
                    .unwrap_or(&ext_to_source[node])
            }));

        let ext_sym_vars_in_source = ext.inputs.iter().filter_map(|&node| match &ext.dag[node] {
            DataVertex::Symbolic(name) => Some((ext_to_source[&node], name)),
            _ => None,
        });

        for (sym, name) in ext_sym_vars_in_source {
            if let Some(sym_in_source) = self.symbolics.get(name) {
                // If source block has symbolic value, then just change node
                // description to store operation from last symbolic value in source
                self.dag[sym] = DataVertex::OpResult(LinInst::Store(name.clone()));
                self.dag.add_edge(*sym_in_source, sym, 0);
            } else {
                // Otherwise keep symbolic input node
                self.inputs.push(sym);
            }

            self.symbolics
                .insert(name.clone(), ext_to_source[&ext.symbolics[name]]);
        }

        self.jump_decided_by = ext.jump_decided_by.as_ref().map(|id| ext_to_source[id]);
        assert!(!algo::is_cyclic_directed(&self.dag));
    }

    fn remove_jump_decision(&mut self) {
        if let Some(jump) = self.jump_decided_by.take() {
            self.dag.remove_node(jump);
        }
    }

    fn remove_stores(&mut self) {
        while let Some(store) = self
            .dag
            .node_indices()
            .find(|&id| matches!(&self.dag[id], DataVertex::OpResult(LinInst::Store(_))))
        {
            eprintln!("Found store: {store:?}");

            let store_source = self
                .dag
                .edges_directed(store, Direction::Incoming)
                .map(|edge| edge.source())
                .collect_vec();

            let store_usages = self
                .dag
                .edges_directed(store, Direction::Outgoing)
                .map(|edge| (edge.target(), *edge.weight()))
                .collect_vec();

            // this loop must have exactly one iteration
            for from in store_source {
                for (to, w) in store_usages.iter().cloned() {
                    self.dag.add_edge(from, to, w);
                }

                self.change_output_state(store, from);
            }

            self.dag.remove_node(store);
        }
    }

    // NOTE: is removal of input node breaks relation of block? probably not,
    // because it's only matters for stack values, but we annotate stack depth
    // for each node.
    // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    // this is actually false. for example we can have following structure
    //
    // A {0, 1} -> B {1} -> C {0}. if B contains dead code dependent on stack variables, then
    // stack variable by itself preserve stack levels used in C. If we remove all information
    // about using {1} in B, then after block merge, C will use {1} not desired {0}.
    //
    // TODO: is some cases symbolic variable may be useless for specific block,
    // but actually bypassed for next blocks
    fn eliminate_dead_code(&mut self) {
        let stack_inputs: BTreeSet<_> = self
            .inputs
            .iter()
            .cloned()
            .filter(|&input| matches!(self.dag[input], DataVertex::StackVar(_)))
            .collect();

        let outputs: BTreeSet<_> = self
            .jump_decided_by
            .iter()
            .chain(self.outputs_nodes().iter())
            .chain(self.symbolics.values())
            .cloned()
            .collect();

        while let Some(sink) = self
            .dag
            .node_indices()
            .filter(|id| !stack_inputs.contains(id))
            .filter(|id| !outputs.contains(id))
            .find(|id| self.dag.edges_directed(*id, Direction::Outgoing).count() == 0)
        {
            self.dag.remove_node(sink);
            self.inputs.retain(|&x| x != sink);
        }
    }

    fn change_output_state(&mut self, node: NodeIndex, new_node: NodeIndex) {
        if let Some((i, _)) = self
            .outputs
            .stack
            .iter()
            .find_position(|&&out_node| out_node == node)
        {
            self.outputs.stack[i] = new_node;
        }

        if self.jump_decided_by.is_some_and(|jmp| jmp == node) {
            self.jump_decided_by = Some(new_node);
        }

        self.symbolics
            .values_mut()
            .filter(|s| **s == node)
            .for_each(|s| *s = new_node);
    }

    /// Replace node with new value and saves *only* outgoing edges from this node.
    /// All incoming edges deleted. Replaces all entries of this node in special nodes (stack outputs,
    /// symbolics, jump decision variable) with new node.
    fn replace_node_for_outgoings(&mut self, node: NodeIndex, new_node: DataVertex) -> NodeIndex {
        let new_node = self.dag.add_node(new_node);

        let outgoing = self
            .dag
            .edges_directed(node, Direction::Outgoing)
            .map(|edge| (edge.target(), *edge.weight()))
            .collect_vec();

        for (to, arg_pos) in outgoing {
            self.dag.add_edge(new_node, to, arg_pos);
        }

        self.change_output_state(node, new_node);
        self.dag.remove_node(node);
        new_node
    }

    fn constant_propagation(&mut self) {
        while let Some((node, op)) = self
            .dag
            .node_references()
            .filter_map(|(i, v)| match v {
                DataVertex::OpResult(LinInst::BinOp(op)) => Some((i, op)),
                _ => None,
            })
            .find(|(i, _)| {
                self.dag
                    .neighbors_directed(*i, Direction::Incoming)
                    .all(|v| matches!(self.dag[v], DataVertex::OpResult(LinInst::Const(_))))
            })
        {
            let Some([lhs, rhs]) = self
                .dag
                .edges_directed(node, Direction::Incoming)
                .sorted_by_key(|edge| edge.weight())
                .rev()
                .filter_map(|edge| match self.dag[edge.source()] {
                    DataVertex::OpResult(LinInst::Const(v)) => Some(v),
                    _ => None,
                })
                .collect_array()
            else {
                break;
            };

            let value = op.eval(lhs, rhs);
            self.replace_node_for_outgoings(node, DataVertex::OpResult(LinInst::Const(value)));
        }
    }

    fn tag_check_evaluation(&mut self) {
        while let Some((node, new_value)) = self
            .dag
            .edge_references()
            .filter_map(
                |edge| match (&self.dag[edge.source()], &self.dag[edge.target()]) {
                    (
                        DataVertex::OpResult(LinInst::SExp(have_tag, have_args)),
                        DataVertex::OpResult(LinInst::Tag(expect_tag, expect_args)),
                    ) => {
                        if have_tag == expect_tag && have_args == expect_args {
                            Some((edge.target(), 1))
                        } else {
                            Some((edge.target(), 0))
                        }
                    }
                    _ => None,
                },
            )
            .next()
        {
            self.replace_node_for_outgoings(node, DataVertex::OpResult(LinInst::Const(new_value)));
        }
    }

    fn optimize(&mut self, flags: DataGraphOptimFlags) {
        if flags.elim_stores {
            self.remove_stores();
        }

        if flags.elim_dead_code {
            self.eliminate_dead_code();
        }

        if flags.const_prop {
            self.constant_propagation();
            self.eliminate_dead_code();
        }

        if flags.tag_eval {
            self.tag_check_evaluation();
            self.eliminate_dead_code();
        }
    }

    fn shift_symbolics_for_inline(&mut self, first_free_loc: u16, args_count: u16) {
        for node in self.dag.node_weights_mut() {
            match node {
                DataVertex::Symbolic(sym) => {
                    *sym = sym_shift(sym.clone(), first_free_loc, args_count)
                }
                _ => {}
            }
        }

        self.symbolics = mem::take(&mut self.symbolics)
            .into_iter()
            .map(|(k, v)| (sym_shift(k, first_free_loc, args_count), v))
            .collect();
    }

    // FIXME: need to somehow recognize and drop unused stack variables.
    fn compile(&self, free_loc: &mut u16) -> Vec<LinInst> {
        fn compile_node(
            node: NodeIndex,
            dag: &StableGraph<DataVertex, ArgStackOffset>,
            symbolics: &BTreeMap<Sym, NodeIndex>,
            already_compiled: &mut BTreeMap<NodeIndex, u16>,
            free_loc: &mut u16,
            code: &mut Vec<LinInst>,
        ) {
            eprintln!("compile: {:?}", dag[node]);

            if let Some(&loc) = already_compiled.get(&node) {
                code.push(LinInst::Load(Sym::Loc(loc)));
                return;
            }

            for dep in dag
                .edges_directed(node, Direction::Incoming)
                .sorted_by_key(|e| e.weight())
                .rev()
            {
                compile_node(
                    dep.source(),
                    dag,
                    symbolics,
                    already_compiled,
                    free_loc,
                    code,
                );
            }

            match &dag[node] {
                DataVertex::Symbolic(sym) => {
                    code.push(LinInst::Load(sym.clone()));
                }
                DataVertex::StackVar(_) => {
                    // NOTE: should be careful here, invariants of analyzed stack machine code
                    // says that stack variable can only appear with "natural" computations, i. e.
                    // doesn't need to shift anything, but can add assertion
                }
                DataVertex::OpResult(inst) => {
                    code.push(inst.clone());
                }
            }

            let used_multiple_times = dag.edges_directed(node, Direction::Outgoing).count() > 1;
            let used_for_store_output_sym = symbolics.values().contains(&node);
            let was_saved = already_compiled.contains_key(&node);

            if (used_multiple_times || used_for_store_output_sym) && !was_saved {
                let save = *free_loc;
                code.push(LinInst::Store(Sym::Loc(save)));
                eprintln!("save loc[{save}] for {:?}", dag[node]);
                *free_loc += 1;
                already_compiled.insert(node, save);
            }
        }

        eprintln!("start compile linear block: {}", self.start_label);

        let output_nodes: BTreeSet<_> = self
            .outputs
            .stack
            .iter()
            .cloned()
            .chain(self.jump_decided_by)
            .collect();

        eprintln!(
            "output nodes: {:?}",
            self.outputs
                .stack
                .iter()
                .cloned()
                .chain(self.jump_decided_by)
                .collect_vec()
        );

        let mut code = Vec::new();
        let mut already_compiled = BTreeMap::new();

        // Dirty hack: load all stack input in local variables to avoid
        // problems with extracting right stack variable at the end of
        // `compile_node` recursion.
        for (input, _) in self
            .inputs
            .iter()
            .filter_map(|&v| match self.dag[v] {
                DataVertex::StackVar(offset) => Some((v, offset)),
                _ => None,
            })
            .sorted_by_key(|(_, offset)| *offset)
        {
            code.push(LinInst::Store(Sym::Loc(*free_loc)));
            code.push(LinInst::Drop);
            already_compiled.insert(input, *free_loc);
            *free_loc += 1;
        }

        // FIXME: need to compile dead nodes early to drop unused stack variables, but
        // this can backfire if unused stack input variable in between used.
        let dead_nodes = self
            .dag
            .node_indices()
            .filter(|v| !output_nodes.contains(v))
            .filter(|&v| self.dag.edges_directed(v, Direction::Outgoing).count() == 0);

        for node in dead_nodes {
            eprintln!("compile dead node: {:?}", self.dag[node]);

            compile_node(
                node,
                &self.dag,
                &self.symbolics,
                &mut already_compiled,
                free_loc,
                &mut code,
            );

            // TODO: Do we need to always do this?
            code.push(LinInst::Drop);
        }

        for node in self
            .outputs
            .stack
            .iter()
            .cloned()
            .chain(self.jump_decided_by)
        {
            eprintln!("compile stack output node: {:?}", self.dag[node]);

            compile_node(
                node,
                &self.dag,
                &self.symbolics,
                &mut already_compiled,
                free_loc,
                &mut code,
            );
        }

        // Store all needed variables only after computations within expression
        // to avoid mess up intermediate references to variables.
        for (sym, node) in self.symbolics.iter() {
            eprintln!("Try to find local tmp value for {sym} in {already_compiled:?}");
            let saved = already_compiled[node];
            code.extend([
                LinInst::Load(Sym::Loc(saved)),
                LinInst::Store(sym.clone()),
                LinInst::Drop,
            ]);
        }

        code
    }
}

fn sym_shift(sym: Sym, first_free_loc: u16, args_count: u16) -> Sym {
    match sym {
        Sym::Arg(i) => Sym::Loc(first_free_loc + i),
        Sym::Loc(i) => Sym::Loc(first_free_loc + args_count + i),
        Sym::Glb(glb) => Sym::Glb(glb),
    }
}

type Label = String;

impl FlowGraph {
    fn args_count(&self) -> u16 {
        self.graph
            .node_weights()
            .filter_map(|v| v.block().map(DataGraph::args_count))
            .max()
            .unwrap_or(0)
    }

    fn loc_count(&self) -> u16 {
        self.graph
            .node_weights()
            .filter_map(|v| v.block().map(DataGraph::locs_count))
            .max()
            .unwrap_or(0)
    }

    fn eliminate_dead_code(&mut self) {
        while let Some(source) = self
            .graph
            .node_indices()
            .filter(|&id| id != self.input)
            .find(|&id| self.graph.edges_directed(id, Direction::Incoming).count() == 0)
        {
            self.graph.remove_node(source);
        }
    }

    fn jump_on_const(&mut self) {
        while let Some((node, v)) = self
            .graph
            .node_references()
            .filter_map(|(id, v)| v.block().map(|block| (id, block)))
            .filter(|(id, _)| {
                self.graph
                    .edges_directed(*id, Direction::Outgoing)
                    .map(|edge| edge.weight())
                    .any(JumpCondition::is_conditional)
            })
            .find_map(|(id, block)| {
                block.jump_decided_by.and_then(|jmp| match block.dag[jmp] {
                    DataVertex::OpResult(LinInst::Const(v)) => Some((id, v)),
                    _ => None,
                })
            })
        {
            if let FlowVertex::LinearBlock(block) = &mut self.graph[node] {
                block.remove_jump_decision();
            }

            let outgoings = self
                .graph
                .edges_directed(node, Direction::Outgoing)
                .map(|edge| (edge.id(), *edge.weight()))
                .collect_vec();

            for (edge, cond) in outgoings {
                if cond.will_jump(v) {
                    self.graph[edge] = JumpCondition::Unconditional;
                } else {
                    self.graph.remove_edge(edge);
                }
            }
        }
    }

    fn merge_block_with_unconditional_jump(&mut self) {
        while let Some((from, to)) = self
            .graph
            .node_references()
            .filter(|(id, vertex)| *id != self.input && vertex.is_block())
            .find_map(|(to, _)| {
                self.graph
                    .edges_directed(to, Direction::Incoming)
                    .map(|edge| (edge.source(), *edge.weight()))
                    .collect_array()
                    .and_then(|x| match x {
                        [(from, JumpCondition::Unconditional)] if self.graph[from].is_block() => {
                            Some((from, to))
                        }
                        _ => None,
                    })
            })
        {
            let ext = if let FlowVertex::LinearBlock(block) = &self.graph[to] {
                block.clone()
            } else {
                unreachable!("by filter")
            };

            if let FlowVertex::LinearBlock(block) = &mut self.graph[from] {
                block.extend(&ext);
            }

            let outgoings = self
                .graph
                .edges_directed(to, Direction::Outgoing)
                .map(|edge| (edge.target(), *edge.weight()))
                .collect_vec();

            for (to, w) in outgoings {
                self.graph.add_edge(from, to, w);
            }

            // Since `from` and `to` is one block, we should preserve output status in merged block
            if self.outputs.contains(&to) {
                self.outputs.retain(|&out| out != from && out != to);
                self.outputs.push(from);
            }

            self.graph.remove_node(to);
        }
    }

    fn optimize(&mut self, flow_optim: FlowOptimFlags) {
        for _ in 0..flow_optim.passes {
            for block in self.graph.node_weights_mut() {
                if let FlowVertex::LinearBlock(graph) = block {
                    graph.optimize(flow_optim.data_flags);
                }
            }

            if flow_optim.elim_dead_code {
                self.eliminate_dead_code();
            }

            if flow_optim.jump_on_const {
                self.jump_on_const();
                self.eliminate_dead_code();
            }

            if flow_optim.merge_blocks {
                self.merge_block_with_unconditional_jump();
            }
        }
    }

    fn analyze_function(ctx: &mut Ctx, code: Vec<Inst>) -> Self {
        let mut edges: Vec<(NodeIndex, Label, JumpCondition)> = Vec::new();
        let mut edge_from_prev: Option<(NodeIndex, JumpCondition)> = None;
        let mut block: Vec<LinInst> = Vec::new();
        let mut graph = StableGraph::new();
        let mut labeled: Option<Label> = None;
        let mut block_indexes = Vec::new(); // only purpose is to find input and outputs blocks
        let mut from_label_to_block: BTreeMap<Label, NodeIndex> = BTreeMap::new();

        // skip(1) for Begin operation
        for (i, inst) in code.into_iter().enumerate().skip(1) {
            match inst {
                Inst::LinInst(inst) => {
                    block.push(inst);
                }
                Inst::FlowInst(inst) => {
                    let start_label = labeled.take().unwrap_or(ctx.fresh_label());
                    let has_cjmp = inst.conditional_jmp();
                    let this_block =
                        analyze_lin_block(start_label.clone(), mem::take(&mut block), has_cjmp);

                    let node = graph.add_node(FlowVertex::LinearBlock(this_block));
                    from_label_to_block.insert(start_label, node);
                    block_indexes.push(node);

                    if let Some((prev, cond)) = edge_from_prev.take() {
                        graph.add_edge(prev, node, cond);
                    }

                    match inst {
                        FlowInst::Jmp(JumpMode::Unconditional, label) => {
                            edges.push((node, label, JumpCondition::Unconditional));
                        }
                        FlowInst::Jmp(jump_mode, label) => {
                            edges.push((node, label, jump_mode.into()));
                            edge_from_prev = Some((node, jump_mode.rev().into()))
                        }
                        FlowInst::Label(label) => {
                            edge_from_prev = Some((node, JumpCondition::Unconditional));
                            labeled = Some(label);
                        }
                        FlowInst::Call(name, args) => {
                            let call_node = graph.add_node(FlowVertex::Call(CallVertex {
                                label: ctx.fresh_label(),
                                name,
                                args,
                            }));
                            graph.add_edge(node, call_node, JumpCondition::Unconditional);
                            edge_from_prev = Some((call_node, JumpCondition::Unconditional));
                        }
                        FlowInst::CallC(args) => {
                            let call_node = graph.add_node(FlowVertex::CallC(CallCVertex {
                                label: ctx.fresh_label(),
                                args,
                            }));
                            graph.add_edge(node, call_node, JumpCondition::Unconditional);
                            edge_from_prev = Some((call_node, JumpCondition::Unconditional));
                        }
                        FlowInst::STI => {
                            let sti_node = graph.add_node(FlowVertex::STI(STIVertex {
                                label: ctx.fresh_label(),
                            }));
                            graph.add_edge(node, sti_node, JumpCondition::Unconditional);
                            edge_from_prev = Some((sti_node, JumpCondition::Unconditional));
                        }
                        FlowInst::STA => {
                            let sta_node = graph.add_node(FlowVertex::STA(STAVertex {
                                label: ctx.fresh_label(),
                            }));
                            graph.add_edge(node, sta_node, JumpCondition::Unconditional);
                            edge_from_prev = Some((sta_node, JumpCondition::Unconditional));
                        }
                        FlowInst::Begin(_) => continue,
                        FlowInst::End => break,
                    }
                }
                Inst::Decl(_) => {}
            }
        }

        for (from, to, cond) in edges {
            graph.add_edge(from, from_label_to_block[&to], cond);
        }

        FlowGraph {
            graph,
            input: block_indexes[0],
            outputs: block_indexes.last().into_iter().cloned().collect(),
        }
    }

    fn replace_node_with_graph(&mut self, node: NodeIndex, replacement: &FlowGraph) {
        assert_ne!(self.input, node, "Can't replace input node with graph");

        for &out in self.outputs.iter() {
            assert_ne!(out, node, "Can't replace output node with graph");
        }

        for edge in self.graph.edges_directed(node, Direction::Outgoing) {
            assert_eq!(
                *edge.weight(),
                JumpCondition::Unconditional,
                "All jumps from replaced node should be unconditional"
            );
        }

        let incomings = self
            .graph
            .edges_directed(node, Direction::Incoming)
            .map(|edge| (edge.source(), *edge.weight()))
            .collect_vec();

        let outgoings = self
            .graph
            .edges_directed(node, Direction::Outgoing)
            .map(|edge| edge.target())
            .collect_vec();

        let replacement_remapping = add_graph(&mut self.graph, &replacement.graph);

        for (from, cond) in incomings {
            self.graph
                .add_edge(from, replacement_remapping[&replacement.input], cond);
        }

        for out in replacement.outputs.iter() {
            let out = replacement_remapping[out];

            for to in outgoings.iter() {
                self.graph.add_edge(out, *to, JumpCondition::Unconditional);
            }
        }

        self.graph.remove_node(node);
    }

    // TODO: how to actually deal with symbolics after inlining?
    // TODO: need to insert additional linear block that will deal
    //  with passing function arguments from stack
    fn replace_call_with_graph(&mut self, call: NodeIndex, mut replacement: FlowGraph) {
        let first_free_loc = self.loc_count();
        let call_args = replacement.args_count();
        replacement.shift_symbolics_for_inline(first_free_loc);

        let prev_input = replacement.input;
        let prologue = gen_inline_call_prologue("prologue".to_string(), call_args, first_free_loc);
        let input = replacement
            .graph
            .add_node(FlowVertex::LinearBlock(prologue));

        replacement.input = input;
        replacement
            .graph
            .add_edge(input, prev_input, JumpCondition::Unconditional);

        self.replace_node_with_graph(call, &replacement);
    }

    fn replace_all_calls(&mut self, call: &str, replacement: &FlowGraph, mut unfold_limit: u32) {
        let expect_args = replacement.args_count();

        while let Some(node) = self.graph.node_references().find_map(|(id, v)| match v {
            FlowVertex::Call(call_v) if call_v.name == call => {
                assert_eq!(call_v.args as u16, expect_args);
                Some(id)
            }
            _ => None,
        }) {
            if unfold_limit == 0 {
                break;
            }

            self.replace_call_with_graph(node, replacement.clone());
            unfold_limit -= 1;
        }
    }

    fn shift_symbolics_for_inline(&mut self, first_free_loc: u16) {
        let args = self.args_count();

        for node in self.graph.node_weights_mut() {
            if let FlowVertex::LinearBlock(block) = node {
                block.shift_symbolics_for_inline(first_free_loc, args);
            }
        }
    }

    fn get_label(&self, node: NodeIndex) -> String {
        match &self.graph[node] {
            FlowVertex::LinearBlock(block) => block.start_label.clone(),
            FlowVertex::Call(call) => call.label.clone(),
            FlowVertex::STI(sti) => sti.label.clone(),
            FlowVertex::STA(sta) => sta.label.clone(),
            FlowVertex::CallC(callc) => callc.label.clone(),
        }
    }

    fn compile(&self, name: &str) -> (Vec<Inst>, u16, u16, u16) {
        let mut code = Vec::new();
        let mut free_loc = self.loc_count();

        fn compile_vertex(
            flow: &FlowGraph,
            node: NodeIndex,
            code: &mut Vec<Inst>,
            free_loc: &mut u16,
        ) {
            code.push(Inst::FlowInst(FlowInst::Label(flow.get_label(node))));

            match &flow.graph[node] {
                FlowVertex::LinearBlock(block) => {
                    code.extend(block.compile(free_loc).into_iter().map(Inst::LinInst));
                }
                FlowVertex::Call(call) => {
                    code.push(Inst::FlowInst(FlowInst::Call(call.name.clone(), call.args)));
                }
                FlowVertex::STI(_) => code.push(Inst::FlowInst(FlowInst::STI)),
                FlowVertex::STA(_) => code.push(Inst::FlowInst(FlowInst::STA)),
                FlowVertex::CallC(callc) => code.push(Inst::FlowInst(FlowInst::CallC(callc.args))),
            }

            // FIXME: dirty hack -- now that there is only one unconditional jump or
            // 2 jumps with different conditions. This loop occurs because of poor
            // choice of graph representation.
            for (i, jmp) in flow
                .graph
                .edges_directed(node, Direction::Outgoing)
                .enumerate()
            {
                let mode = match jmp.weight() {
                    _ if i > 0 => JumpMode::Unconditional,
                    JumpCondition::Unconditional => JumpMode::Unconditional,
                    JumpCondition::Zero => JumpMode::Zero,
                    JumpCondition::NonZero => JumpMode::NonZero,
                };

                code.push(Inst::FlowInst(FlowInst::Jmp(
                    mode,
                    flow.get_label(jmp.target()),
                )));
            }
        }

        eprintln!("start compile function: {name}");

        compile_vertex(self, self.input, &mut code, &mut free_loc);

        for vertex in self
            .graph
            .node_indices()
            .filter(|v| *v != self.input)
            .filter(|v| !self.outputs.contains(v))
        {
            compile_vertex(self, vertex, &mut code, &mut free_loc);
        }

        // FIXME: change to get fresh label?
        let exit_label = format!("{name}_exit");

        // FIXME: rethink case when output block is input block at the same time
        for &output in self.outputs.iter().filter(|&&output| output != self.input) {
            compile_vertex(self, output, &mut code, &mut free_loc);
            code.push(Inst::FlowInst(FlowInst::Jmp(
                JumpMode::Unconditional,
                exit_label.clone(),
            )));
        }

        code.push(Inst::FlowInst(FlowInst::Label(exit_label)));
        (code, self.args_count(), free_loc, 0)
    }
}

fn write_code<W: Write>(w: &mut W, code: &[Inst]) -> fmt::Result {
    for inst in code.iter() {
        writeln!(w, "{inst}")?;
    }

    writeln!(w, "!!")?;

    Ok(())
}

fn gen_inline_call_prologue(label: String, args: u16, first_free_loc: u16) -> DataGraph {
    let inst = (0..args)
        .rev()
        .flat_map(|i| [LinInst::Store(Sym::Loc(first_free_loc + i)), LinInst::Drop])
        .collect_vec();

    analyze_lin_block(label, inst, false)
}

struct UnitOptimFlags {
    flow_optim: FlowOptimFlags,
    force_inline: Vec<String>,
}

#[derive(Debug)]
struct Unit {
    functions: BTreeMap<Ident, FlowGraph>,
    declarations: Vec<Decl>,
}

impl Unit {
    fn analyze(ctx: &mut Ctx, code: impl Iterator<Item = Inst>) -> Unit {
        let mut functions: BTreeMap<Ident, FlowGraph> = BTreeMap::new();
        let mut declarations = Vec::new();
        let mut code = code
            .filter(|inst| match inst {
                Inst::Decl(decl) => {
                    declarations.push(decl.clone());
                    false
                }
                _ => true,
            })
            .peekable();

        while let Some(FlowInst(FlowInst::Label(name))) = code.peek() {
            let name = name.clone();
            let code: Vec<_> = (&mut code)
                .skip(1)
                .take_while_inclusive(|inst| inst != &FlowInst(FlowInst::End))
                .collect();

            functions.insert(name, FlowGraph::analyze_function(ctx, code));
        }

        Unit {
            functions,
            declarations,
        }
    }

    fn optimize(&mut self, flags: UnitOptimFlags) {
        self.functions
            .values_mut()
            .for_each(|flow| flow.optimize(flags.flow_optim));

        for call in flags.force_inline.iter() {
            let call_graph = self.functions[call].clone();

            self.functions.values_mut().for_each(|flow| {
                flow.replace_all_calls(&call, &call_graph, 1);
                flow.optimize(flags.flow_optim);
            });
        }
    }

    fn compile(&self) -> Vec<Inst> {
        let mut code = self.declarations.iter().cloned().map(Decl).collect_vec();

        for (name, function) in self
            .functions
            .iter()
            // FIXME: "main" function should go first in code generation
            // actually should find it in `self.declarations`
            .sorted_by_key(|(name, _)| !name.contains("init"))
        {
            let (body, args, locs, clos) = function.compile(&name);
            code.push(Inst::FlowInst(FlowInst::Label(name.clone())));
            code.push(Inst::FlowInst(FlowInst::Begin(Begin {
                name: name.clone(),
                args: args as u32,
                locs: locs as u32,
                clos: clos as u32,
            })));
            code.extend(body);
            code.push(Inst::FlowInst(FlowInst::End));
        }

        code
    }
}

struct Escaper<W>(W);

impl<W> fmt::Write for Escaper<W>
where
    W: fmt::Write,
{
    fn write_str(&mut self, s: &str) -> fmt::Result {
        for c in s.chars() {
            self.write_char(c)?;
        }

        Ok(())
    }

    fn write_char(&mut self, c: char) -> fmt::Result {
        match c {
            '"' | '\\' => self.0.write_char('\\')?,

            // \l is for left justified linebreak
            '\n' => return self.0.write_str("\\l"),

            _ => {}
        }

        self.0.write_char(c)
    }
}

fn subgraph<W: Write>(w: &mut W, label: &str, graph: &DataGraph) -> fmt::Result {
    writeln!(w, "subgraph cluster_{label} {{")?;
    writeln!(w, "label = \"{}\";", graph.info_label())?;
    writeln!(w, "style = rounded;")?;
    writeln!(w, "color = black;")?;
    writeln!(w, "sub{label}_input [shape = point style = invis];")?;
    writeln!(w, "sub{label}_output [shape = point style = invis];")?;

    for node in graph.inputs.iter() {
        writeln!(w, "sub{label}_input -> sub{label}{};", node.index())?;
    }

    for node in graph.outputs_nodes().iter() {
        writeln!(w, "sub{label}{} -> sub{label}_output;", node.index())?;
    }

    for node in graph.dag.node_indices() {
        let node_symbolics = graph
            .symbolics
            .iter()
            .filter(|(_, v)| node == **v)
            .map(|(sym, _)| sym)
            .collect_vec();

        write!(w, "sub{label}{} [label = \"", node.index(),)?;
        write!(
            Escaper(&mut *w),
            "{:?}; {node_symbolics:?}",
            &graph.dag[node]
        )?;
        writeln!(
            w,
            "\" {}];",
            match () {
                _ if graph.inputs.contains(&node) && graph.outputs_nodes().contains(&node) =>
                    "color = yellow",
                _ if graph.inputs.contains(&node) => "color = green",
                _ if graph.outputs_nodes().contains(&node) => "color = red",
                _ if graph.jump_decided_by.iter().contains(&node) => "color = purple",
                _ => "",
            }
        )?;
    }

    for edge in graph.dag.edge_references() {
        writeln!(
            w,
            "sub{label}{} -> sub{label}{} [label = \"{}\"];",
            edge.source().index(),
            edge.target().index(),
            edge.weight()
        )?;
    }

    writeln!(w, "}}")?;
    Ok(())
}

fn call_subgraph<W: Write>(w: &mut W, label: &str, name: &str, args: usize) -> fmt::Result {
    writeln!(w, "subgraph cluster_{label} {{")?;
    writeln!(w, "label = \"call ({}, {})\";", name, args)?;
    writeln!(w, "style = filled;")?;
    writeln!(w, "color = lightgrey;")?;
    writeln!(w, "sub{label}_input [shape = point style = invis];")?;
    writeln!(w, "sub{label}_output [shape = point style = invis];")?;
    writeln!(w, "}}")?;
    Ok(())
}

fn callc_subgraph<W: Write>(w: &mut W, label: &str, args: usize) -> fmt::Result {
    writeln!(w, "subgraph cluster_{label} {{")?;
    writeln!(w, "label = \"callc {}\";", args)?;
    writeln!(w, "style = filled;")?;
    writeln!(w, "color = lightgrey;")?;
    writeln!(w, "sub{label}_input [shape = point style = invis];")?;
    writeln!(w, "sub{label}_output [shape = point style = invis];")?;
    writeln!(w, "}}")?;
    Ok(())
}

fn sti_subgraph<W: Write>(w: &mut W, label: &str) -> fmt::Result {
    writeln!(w, "subgraph cluster_{label} {{")?;
    writeln!(w, "label = \"STI\";")?;
    writeln!(w, "style = filled;")?;
    writeln!(w, "color = lightgrey;")?;
    writeln!(w, "sub{label}_input [shape = point style = invis];")?;
    writeln!(w, "sub{label}_output [shape = point style = invis];")?;
    writeln!(w, "}}")?;
    Ok(())
}

fn sta_subgraph<W: Write>(w: &mut W, label: &str) -> fmt::Result {
    writeln!(w, "subgraph cluster_{label} {{")?;
    writeln!(w, "label = \"STA\";")?;
    writeln!(w, "style = filled;")?;
    writeln!(w, "color = lightgrey;")?;
    writeln!(w, "sub{label}_input [shape = point style = invis];")?;
    writeln!(w, "sub{label}_output [shape = point style = invis];")?;
    writeln!(w, "}}")?;
    Ok(())
}

fn function_graph<W: Write>(w: &mut W, flow: &FlowGraph) -> fmt::Result {
    writeln!(w, "digraph G {{")?;
    writeln!(w, "compound = true;")?;
    writeln!(w, "node [shape=box]")?;

    for block in flow.graph.node_indices() {
        let label = block.index().to_string();
        match &flow.graph[block] {
            FlowVertex::LinearBlock(graph) => subgraph(w, &label, graph)?,
            FlowVertex::Call(call) => call_subgraph(w, &label, &call.name, call.args)?,
            FlowVertex::STI(_) => sti_subgraph(w, &label)?,
            FlowVertex::STA(_) => sta_subgraph(w, &label)?,
            FlowVertex::CallC(callc) => callc_subgraph(w, &label, callc.args)?,
        }
    }

    for edge in flow.graph.edge_references() {
        writeln!(
            w,
            "sub{}_output -> sub{}_input [label = \"{:?}\" ltail = cluster_{} lhead=cluster_{}]",
            edge.source().index(),
            edge.target().index(),
            edge.weight(),
            edge.source().index(),
            edge.target().index(),
        )?;
    }

    writeln!(w, "}}")?;
    Ok(())
}

fn parse_stack_code(code: &str) -> Vec<Inst> {
    code.lines()
        .filter(|line| !line.chars().all(char::is_whitespace))
        .filter(|line| !line.contains("META")) // FIXME: process failed pattern matching
        .map(|line| Inst::parse(line).expect(&format!("Failed to parse: {line}")))
        .collect()
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Filepath to stack machine code (.sm)
    #[arg(short, long)]
    source: PathBuf,

    /// Path to output dir
    #[arg(short, long, default_value = None)]
    graphs_dir: Option<PathBuf>,

    /// Eliminate dead code
    #[arg(short, long, default_value_t = false)]
    elim_dead_code: bool,

    #[arg(long, default_value_t = true)]
    elim_stores: bool,

    /// Propagate constant
    #[arg(short, long, default_value_t = false)]
    const_prop: bool,

    /// Replace tag check on known tag with constant
    #[arg(short, long, default_value_t = false)]
    tag_check_eval: bool,

    /// Replace conditional jump on constant with unconditional jump
    #[arg(short, long, default_value_t = false)]
    jump_on_const: bool,

    /// Merge blocks with unconditional jump between them
    #[arg(short, long, default_value_t = false)]
    merge_blocks: bool,

    /// Try optimize flow with `passes` iterations
    #[arg(short, long, default_value_t = 1)]
    passes: u32,

    #[arg(long, value_delimiter = ' ', num_args = 1.., default_value = None)]
    force_inline: Option<Vec<String>>,

    #[arg(short = 'O', long, default_value_t = false)]
    optim_full: bool,
}

fn data_flags_from_args(args: &Args) -> DataGraphOptimFlags {
    if args.optim_full {
        DataGraphOptimFlags {
            elim_dead_code: true,
            const_prop: true,
            tag_eval: true,
            elim_stores: true,
        }
    } else {
        DataGraphOptimFlags {
            elim_dead_code: args.elim_dead_code,
            const_prop: args.const_prop,
            tag_eval: args.tag_check_eval,
            elim_stores: args.elim_stores,
        }
    }
}

fn flow_flags_from_args(args: &Args) -> FlowOptimFlags {
    let data_flags = data_flags_from_args(args);

    if args.optim_full {
        FlowOptimFlags {
            elim_dead_code: true,
            jump_on_const: true,
            data_flags,
            merge_blocks: true,
            passes: args.passes,
        }
    } else {
        FlowOptimFlags {
            elim_dead_code: args.elim_dead_code,
            jump_on_const: args.jump_on_const,
            data_flags,
            merge_blocks: args.merge_blocks,
            passes: args.passes,
        }
    }
}

fn unit_flags_from_args(args: &Args) -> UnitOptimFlags {
    UnitOptimFlags {
        flow_optim: flow_flags_from_args(args),
        force_inline: args
            .force_inline
            .as_ref()
            .map(|x| x.clone())
            .unwrap_or_default(),
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let flags = unit_flags_from_args(&args);
    let content = read_to_string(File::open(&args.source)?)?;
    // eprintln!("Get SM code at rust side:");
    // eprintln!("{content}");
    let code = parse_stack_code(&content);
    let mut ctx = Ctx::default();
    let mut unit = Unit::analyze(&mut ctx, code.into_iter());
    unit.optimize(flags);

    if let Some(out_dir) = args.graphs_dir {
        for (name, flow_graph) in unit.functions.iter() {
            let mut buffer = String::new();
            function_graph(&mut buffer, flow_graph)?;
            fs::write(out_dir.join(name).with_extension("dot"), buffer)?;
        }
    }

    let mut buffer = String::new();
    write_code(&mut buffer, &unit.compile())?;
    fs::write(args.source.with_extension("osm"), buffer)?;

    Ok(())
}
