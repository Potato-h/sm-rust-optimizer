use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{self, Write};
use std::fs::{self, File};
use std::io::read_to_string;
use std::path::PathBuf;
use std::{iter, mem};

use clap::Parser;
use either::Either::{self, Left, Right};
use itertools::Itertools;
use petgraph::csr::IndexType;
use petgraph::visit::{EdgeRef, IntoEdgeReferences, IntoEdgesDirected, IntoNodeReferences};
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
    Eq,
    Gt,
}

impl Op {
    fn eval(&self, lhs: i32, rhs: i32) -> i32 {
        match self {
            Op::Plus => lhs + rhs,
            Op::Minus => lhs - rhs,
            Op::Mul => lhs * rhs,
            Op::Eq => (lhs == rhs) as i32,
            Op::Gt => (lhs > rhs) as i32,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum LinInst {
    Const(i32),
    Elem,
    Store(Ident),
    Load(Ident),
    BinOp(Op),
    Tag(Ident, usize),
    SExp(Ident, usize),
    Dup,
    Drop,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum FlowInst {
    Jmp(JumpMode, Ident),
    Label(Ident),
    // call command should be in flow graph, because it can cause
    // arbitrary change of global variables. And after unfolding it can
    // have control flow instruction which weird to extract from lin block
    Call(Ident, usize),
    Ret,
    Begin(Ident),
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

#[derive(Debug, Clone, Copy)]
struct DataGraphOptimFlags {
    elim_dead_code: bool,
    const_prop: bool,
    tag_eval: bool,
}

#[derive(Debug, Clone, Copy)]
struct FlowOptimFlags {
    elim_dead_code: bool,
    jump_on_const: bool,
    data_flags: DataGraphOptimFlags,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Inst {
    LinInst(LinInst),
    FlowInst(FlowInst),
}

impl Inst {
    fn parse(code: &str) -> Option<Inst> {
        let mut tokens = code
            .split(|ch| match ch {
                ',' | '(' | ')' => true,
                _ if ch.is_whitespace() => true,
                _ => false,
            })
            .filter(|token| !token.is_empty());

        match tokens.next()? {
            "BEGIN" => {
                let name = tokens.next()?;
                Some(FlowInst(FlowInst::Begin(name.to_string())))
            }
            "END" => Some(FlowInst(FlowInst::End)),
            "LD" => {
                let place = tokens.next()?.to_string();
                Some(LinInst(LinInst::Load(place)))
            }
            "ST" => {
                let place = tokens.next()?.to_string();
                Some(LinInst(LinInst::Store(place)))
            }
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
            "BINOP" => match tokens.next() {
                Some("+") => Some(LinInst(LinInst::BinOp(Op::Plus))),
                Some("-") => Some(LinInst(LinInst::BinOp(Op::Minus))),
                Some("*") => Some(LinInst(LinInst::BinOp(Op::Mul))),
                Some("==") => Some(LinInst(LinInst::BinOp(Op::Eq))),
                Some(">") => Some(LinInst(LinInst::BinOp(Op::Gt))),
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
            _ => panic!("unknown command"),
        }
    }
}

type ArgStackOffset = usize;

// TODO: aside from outputs on stack, should store symbolics
#[derive(Debug, Clone)]
struct DataGraph {
    start_label: String,
    dag: StableGraph<DataVertex, ArgStackOffset>,
    inputs: Vec<NodeIndex>,
    outputs: Vec<NodeIndex>,
    symbolics: BTreeMap<Ident, NodeIndex>,
    jump_decided_by: Option<NodeIndex>,
}

impl DataGraph {
    fn info_label(&self) -> String {
        format!(
            "{}\\linputs: {}\\loutputs: {}",
            self.start_label,
            self.inputs.len(),
            self.outputs.len()
        )
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
    Symbolic(Ident),
    StackVar(usize),
    OpResult(LinInst),
}

#[derive(Debug)]
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
enum FlowVertex {
    LinearBlock(DataGraph),
    Call(String, usize),
}

#[derive(Debug)]
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
    let mut symbolics: BTreeMap<Ident, NodeIndex> = BTreeMap::new();

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
            LinInst::Elem => n_arity_inst(&mut dag, &mut stack, 2, inst),
            LinInst::Store(ref name) => {
                let node = dag.add_node(inst_vertex(&inst));
                let from = stack.peek().right_or_else(|var| dag.add_node(var));
                dag.add_edge(from, node, 0);
                symbolics.insert(name.clone(), node);
            }
            LinInst::Load(ref name) => {
                if !symbolics.contains_key(name) {
                    let node = dag.add_node(DataVertex::Symbolic(name.clone()));
                    symbolics.insert(name.clone(), node);
                }

                stack.push(symbolics[name]);
            }
            LinInst::BinOp(_) => n_arity_inst(&mut dag, &mut stack, 2, inst),
            LinInst::Tag(_, _) => n_arity_inst(&mut dag, &mut stack, 1, inst),
            LinInst::SExp(_, n) => n_arity_inst(&mut dag, &mut stack, n, inst),
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
                stack.pop();
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
        outputs: stack.stack,
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
    fn remove_jump_decision(&mut self) {
        if let Some(jump) = self.jump_decided_by.take() {
            self.dag.remove_node(jump);
        }
    }

    // TODO: is removal of input node breaks relation of block? probably not,
    // because it's only matters for stack values, but we annotate stack depth
    // for each node.
    // TODO: is some cases symbolic variable may be useless for specific block,
    // but actually bypassed for next blocks
    fn eliminate_dead_code(&mut self) {
        let outputs: BTreeSet<_> = self
            .jump_decided_by
            .iter()
            .chain(self.outputs.iter())
            .cloned()
            .collect();

        while let Some(sink) = self
            .dag
            .node_indices()
            .filter(|id| !outputs.contains(id))
            .find(|id| self.dag.edges_directed(*id, Direction::Outgoing).count() == 0)
        {
            self.dag.remove_node(sink);
            self.inputs.retain(|&x| x != sink);
        }
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

        if let Some((i, _)) = self
            .outputs
            .iter()
            .find_position(|&&out_node| out_node == node)
        {
            self.outputs[i] = new_node;
        }

        if self.jump_decided_by.is_some_and(|jmp| jmp == node) {
            self.jump_decided_by = Some(new_node);
        }

        self.symbolics
            .values_mut()
            .filter(|s| **s == node)
            .for_each(|s| *s = new_node);

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
}

type Label = String;

// TODO: add optimization that will merge block A and B, with B only have unconditional jump from A
impl FlowGraph {
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
            .filter_map(|(id, block)| match block {
                FlowVertex::LinearBlock(block) => Some((id, block)),
                FlowVertex::Call(_, _) => None,
            })
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
            match &mut self.graph[node] {
                FlowVertex::LinearBlock(block) => block.remove_jump_decision(),
                FlowVertex::Call(_, _) => {}
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

    fn optimize(&mut self, flow_optim: FlowOptimFlags) {
        for block in self.graph.node_weights_mut() {
            match block {
                FlowVertex::LinearBlock(graph) => graph.optimize(flow_optim.data_flags),
                FlowVertex::Call(_, _) => {}
            }
        }

        if flow_optim.elim_dead_code {
            self.eliminate_dead_code();
        }

        if flow_optim.jump_on_const {
            self.jump_on_const();
            self.eliminate_dead_code();
        }
    }

    // TODO: fill all outputs (is RET operation a early return or something else?)
    fn analyze_function(code: Vec<Inst>) -> Self {
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
                    let start_label = labeled.take().unwrap_or(format!("Line_{i}"));
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
                            let call_node = graph.add_node(FlowVertex::Call(name, args));
                            graph.add_edge(node, call_node, JumpCondition::Unconditional);
                            edge_from_prev = Some((call_node, JumpCondition::Unconditional));
                        }
                        FlowInst::Ret => todo!(),
                        FlowInst::Begin(_) => continue,
                        FlowInst::End => break,
                    }
                }
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
    }

    // TODO: how to actually deal with symbolics after inlining?
    // TODO: need to insert additional linear block that will deal
    //  with passing function arguments from stack
    fn replace_call_with_graph(&mut self, call: Ident, replacement: &FlowGraph) {
        todo!()
    }
}

#[derive(Debug)]
struct Unit {
    functions: BTreeMap<Ident, FlowGraph>,
}

impl Unit {
    fn analyze(mut insts: impl Iterator<Item = Inst>) -> Unit {
        let mut functions: BTreeMap<Ident, FlowGraph> = BTreeMap::new();

        while let Some(FlowInst(FlowInst::Begin(name))) = insts.next() {
            let code: Vec<_> = iter::once(FlowInst(FlowInst::Begin(name.clone())))
                .chain((&mut insts).take_while_inclusive(|inst| inst != &FlowInst(FlowInst::End)))
                .collect();

            functions.insert(name, FlowGraph::analyze_function(code));
        }

        Unit { functions }
    }

    fn optimize(&mut self, flags: FlowOptimFlags) {
        self.functions
            .values_mut()
            .for_each(|flow| flow.optimize(flags));
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
    writeln!(w, "style = filled;")?;
    writeln!(w, "color = lightgrey;")?;
    writeln!(w, "sub{label}_input [shape = point style = invis];")?;
    writeln!(w, "sub{label}_output [shape = point style = invis];")?;

    for node in graph.inputs.iter() {
        writeln!(w, "sub{label}_input -> sub{label}{};", node.index())?;
    }

    for node in graph.outputs.iter() {
        writeln!(w, "sub{label}{} -> sub{label}_output;", node.index())?;
    }

    for node in graph.dag.node_indices() {
        write!(w, "sub{label}{} [label = \"", node.index(),)?;
        write!(Escaper(&mut *w), "{:?}", &graph.dag[node])?;
        writeln!(
            w,
            "\" {}];",
            match () {
                _ if graph.inputs.contains(&node) && graph.outputs.contains(&node) =>
                    "color = yellow",
                _ if graph.inputs.contains(&node) => "color = green",
                _ if graph.outputs.contains(&node) => "color = red",
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

fn function_graph<W: Write>(w: &mut W, flow: &FlowGraph) -> fmt::Result {
    writeln!(w, "digraph G {{")?;
    writeln!(w, "compound = true;")?;

    for block in flow.graph.node_indices() {
        let label = block.index().to_string();
        match &flow.graph[block] {
            FlowVertex::LinearBlock(graph) => subgraph(w, &label, graph)?,
            FlowVertex::Call(name, args) => call_subgraph(w, &label, name, *args)?,
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

    /// Propagate constant
    #[arg(short, long, default_value_t = false)]
    const_prop: bool,

    /// Replace tag check on known tag with constant
    #[arg(short, long, default_value_t = false)]
    tag_check_eval: bool,

    /// Replace conditional jump on constant with unconditional jump
    #[arg(short, long, default_value_t = false)]
    jump_on_const: bool,

    #[arg(short = 'O', long, default_value_t = false)]
    optim_full: bool,
}

fn data_flags_from_args(args: &Args) -> DataGraphOptimFlags {
    if args.optim_full {
        DataGraphOptimFlags {
            elim_dead_code: true,
            const_prop: true,
            tag_eval: true,
        }
    } else {
        DataGraphOptimFlags {
            elim_dead_code: args.elim_dead_code,
            const_prop: args.const_prop,
            tag_eval: args.tag_check_eval,
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
        }
    } else {
        FlowOptimFlags {
            elim_dead_code: args.elim_dead_code,
            jump_on_const: args.jump_on_const,
            data_flags,
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let flow_flags = flow_flags_from_args(&args);
    let content = read_to_string(File::open(args.source)?)?;
    let code = parse_stack_code(&content);
    let mut unit = Unit::analyze(code.into_iter());
    unit.optimize(flow_flags);

    if let Some(out_dir) = args.graphs_dir {
        for (name, flow_graph) in unit.functions.iter() {
            let mut buffer = String::new();
            function_graph(&mut buffer, flow_graph)?;
            fs::write(out_dir.join(name).with_extension("dot"), buffer)?;
        }
    }

    Ok(())
}
