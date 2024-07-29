use core::fmt;
use std::{env, fs, ops};

use anyhow::bail;
use metadata::tflite::{AssociatedFile, ModelMetadata, ProcessUnit, TensorMetadata};
use object_detector::mediapipe::tasks::ObjectDetectorOptions;

#[allow(warnings)]
#[path = "../generated/object_detector_metadata_schema_generated.rs"]
mod object_detector;

#[allow(warnings)]
#[path = "../generated/metadata_schema_generated.rs"]
mod metadata;

#[allow(warnings)]
#[path = "../generated/schema_v3c_generated.rs"]
mod v3c;

fn main() -> anyhow::Result<()> {
    let args = env::args_os().skip(1).collect::<Vec<_>>();
    let tflite = match &*args {
        [path] => fs::read(path)?,
        _ => bail!("usage: tflite-metadump <model.tflite>"),
    };

    let tflite = flatbuffers::root::<v3c::tflite::Model>(&tflite)?;
    let Some(metadata) = tflite.metadata() else {
        bail!("model contains no metadata entries")
    };

    let Some(metadata) = metadata
        .iter()
        .find(|meta| meta.name() == Some("TFLITE_METADATA"))
    else {
        bail!("model contains no `TFLITE_METADATA` entry")
    };

    let Some(buffers) = tflite.buffers() else {
        bail!("model contains no buffer list")
    };

    let buffer = buffers.get(metadata.buffer() as usize);
    let metadata_bytes = buffer.data().map_or(&[][..], |v| v.bytes());

    let meta = flatbuffers::root::<ModelMetadata>(&metadata_bytes)?;
    if let Some(name) = meta.name() {
        println!("name: {name}");
    }
    if let Some(description) = meta.description() {
        println!("description: {description}");
    }
    if let Some(version) = meta.version() {
        println!("version: {version}");
    }
    if let Some(author) = meta.author() {
        println!("author: {author}");
    }
    if let Some(license) = meta.license() {
        println!("license: {license}");
    }
    if let Some(min_parser_version) = meta.min_parser_version() {
        println!("min_parser_version: {min_parser_version}");
    }
    if let Some(assoc) = meta.associated_files() {
        println!("{} associated file(s):", assoc.len());
        print_assoc_files(Indent(0), assoc.iter());
    }
    if let Some(subs) = meta.subgraph_metadata() {
        println!("{} subgraph(s)", subs.len());
        for sub in subs.iter() {
            println!("- name: {}", sub.name().unwrap_or("<unnamed>"));
            if let Some(desc) = sub.description() {
                println!("  description: {desc}");
            }
            if let Some(assoc) = meta.associated_files() {
                println!("  {} associated file(s):", assoc.len());
                print_assoc_files(Indent(1), assoc.iter());
            }
            if let Some(inp) = sub.input_tensor_metadata() {
                println!("  - {} input tensor(s) with metadata", inp.len());
                print_tensor_meta(Indent(2), inp.iter());
            }
            if let Some(outp) = sub.output_tensor_metadata() {
                println!("  - {} output tensor(s) with metadata", outp.len());
                print_tensor_meta(Indent(2), outp.iter());
            }
            if let Some(inp) = sub.input_process_units() {
                println!("  - {} input tensor process units", inp.len());
                print_proc(Indent(2), inp.iter());
            }
            if let Some(outp) = sub.output_process_units() {
                println!("  - {} output tensor process units", outp.len());
                print_proc(Indent(2), outp.iter());
            }
            if let Some(groups) = sub.input_tensor_groups() {
                println!("  - {} input tensor groups", groups.len());
                for group in groups.iter() {
                    println!("    - {group:?}");
                }
            }
            if let Some(groups) = sub.output_tensor_groups() {
                println!("  - {} output tensor groups", groups.len());
                for group in groups.iter() {
                    println!("    - {group:?}");
                }
            }
            if let Some(custom) = sub.custom_metadata() {
                println!("  - {} custom metadata entries", custom.len());
                for custom in custom.iter() {
                    let bytes = custom.data().map_or(&[][..], |v| v.bytes());
                    println!("    - name: {}", custom.name().unwrap_or("<unnamed>"));
                    println!("      {} bytes", bytes.len());

                    match custom.name() {
                        Some("DETECTOR_METADATA") => {
                            if let Err(e) = decode_and_print_detector_metadata(Indent(3), bytes) {
                                println!("      decoding error: {e}");
                            }
                        }
                        _ => println!("      (unknown or unhandled format)"),
                    }
                }
            }
        }
    }

    Ok(())
}

fn decode_and_print_detector_metadata(indent: Indent, bytes: &[u8]) -> anyhow::Result<()> {
    let options = flatbuffers::root::<ObjectDetectorOptions>(bytes)?;
    if let Some(min) = options.min_parser_version() {
        println!("{indent}min_parser_version: {min}");
    }
    if let Some(dec) = options.tensors_decoding_options() {
        println!("{indent}tensors_decoding_options: {dec:?}");
    }
    if let Some(ssd) = options.ssd_anchors_options() {
        println!("{indent}ssd_anchor_options:");
        if let Some(fixed) = ssd.fixed_anchors_schema() {
            println!("{indent}  fixed_anchors_schema:");
            if let Some(anchors) = fixed.anchors() {
                // (this can contain thousands of anchors, so don't print all of them)
                const PRINT: usize = 24;
                println!("{indent}    {} anchors", anchors.len());
                if anchors.len() <= PRINT {
                    for anchor in anchors.iter() {
                        println!("{indent}    - {anchor:?}");
                    }
                } else {
                    for anchor in anchors.iter().take(PRINT / 2) {
                        println!("{indent}    - {anchor:?}");
                    }
                    println!("{indent}    - ...{} more", anchors.len() - PRINT);
                    for i in anchors.len() - 1 - PRINT / 2..anchors.len() {
                        let anchor = anchors.get(i);
                        println!("{indent}    - {anchor:?}");
                    }
                }
            }
        }
    }

    Ok(())
}

fn print_tensor_meta<'a>(indent: Indent, meta: impl Iterator<Item = TensorMetadata<'a>>) {
    for tens in meta {
        println!("{indent}- name: {}", tens.name().unwrap_or("<unnamed>"));
        if let Some(desc) = tens.description() {
            println!("{indent}  description: {desc}");
        }
        if let Some(dims) = tens.dimension_names() {
            println!("{indent}  dimension names: {dims:?}");
        }
        if let Some(cont) = tens.content() {
            println!("{indent}  content: {cont:?}");
        }
        if let Some(stats) = tens.stats() {
            println!("{indent}  stats: {stats:?}");
        }
        if let Some(assoc) = tens.associated_files() {
            println!("{indent}  {} associated file(s):", assoc.len());
            print_assoc_files(indent + 1, assoc.iter());
        }
        if let Some(proc) = tens.process_units() {
            println!("{indent}  {} process unit(s):", proc.len());
            print_proc(indent + 1, proc.iter());
        }
    }
}

fn print_assoc_files<'a>(indent: Indent, assoc: impl Iterator<Item = AssociatedFile<'a>>) {
    for file in assoc {
        println!("{indent}- name: {}", file.name().unwrap_or("<unnamed>"));
        if let Some(desc) = file.description() {
            println!("{indent}  description: {desc}");
        }
        println!("{indent}  type: {:?}", file.type_());
        if let Some(locale) = file.locale() {
            println!("{indent}  locale: {locale}");
        }
        if let Some(version) = file.version() {
            println!("{indent}  version: {version}");
        }
    }
}

fn print_proc<'a>(indent: Indent, proc: impl Iterator<Item = ProcessUnit<'a>>) {
    for pu in proc {
        println!("{indent}- {pu:?}");
    }
}

#[derive(Copy, Clone)]
struct Indent(usize);

impl ops::Add<usize> for Indent {
    type Output = Self;

    fn add(self, rhs: usize) -> Self::Output {
        Self(self.0 + rhs)
    }
}

impl fmt::Display for Indent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&"  ".repeat(self.0))
    }
}
