use itertools::Itertools;
use petgraph::visit::{EdgeRef, IntoEdgeReferences};
use std::fmt::{self, Display, Write};

use crate::core::{DataGraph, FlowGraph, FlowVertex};

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

struct SepByComma<'a, T>(&'a [T]);

impl<T: Display> Display for SepByComma<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut iter = self.0.iter();

        if let Some(v) = iter.next() {
            write!(f, "{v}")?;
        }

        for v in iter {
            write!(f, ", {v}")?;
        }

        Ok(())
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
            "{}; [{}]",
            &graph.dag[node],
            SepByComma(node_symbolics.as_slice())
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

fn single_node_subgraph<W: Write>(
    w: &mut W,
    label: &str,
    content: impl FnOnce(&mut W) -> fmt::Result,
) -> fmt::Result {
    writeln!(w, "subgraph cluster_{label} {{")?;
    write!(w, "label = \"")?;
    content(w)?;
    writeln!(w, "\";")?;
    writeln!(w, "sub{label}_input [shape = point style = invis];")?;
    writeln!(w, "sub{label}_output [shape = point style = invis];")?;
    writeln!(w, "}}")?;
    Ok(())
}

pub fn function_graph<W: Write>(w: &mut W, flow: &FlowGraph) -> fmt::Result {
    writeln!(w, "digraph G {{")?;
    writeln!(w, "compound = true;")?;
    writeln!(w, "node [shape=box]")?;

    for block in flow.graph.node_indices() {
        let label = block.index().to_string();
        match &flow.graph[block] {
            FlowVertex::LinearBlock(graph) => subgraph(w, &label, graph)?,
            FlowVertex::Call(call) => single_node_subgraph(w, &label, |w| {
                write!(w, "CALL {}, {}", call.name, call.args)
            })?,
            FlowVertex::STI(_) => single_node_subgraph(w, &label, |w| write!(w, "STI"))?,
            FlowVertex::STA(_) => single_node_subgraph(w, &label, |w| write!(w, "STA"))?,
            FlowVertex::CallC(callc) => {
                single_node_subgraph(w, &label, |w| write!(w, "CALLC {}", callc.args))?
            }
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
