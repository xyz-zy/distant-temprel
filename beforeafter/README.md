# BeforeAfter Example Parser

1. Generate syntactic parse tree and output to `TREE_DIR/<filename>_<rel>.tree/txt`.

2. Process parse tree and extract event-event relation. Output to `EXAMPLE_DIR/<filename>_<rel>.json`.

At run-time: load examples from `EXAMPLE_DIR`.


## Files

`filter_xml.py --xml_path 'XML_DIR/*.xml' --out_dir TREE_DIR/`

    Filters parse trees in XML to ones containing before/after mentions.

 * `--xml_path`: Will be glob'd, every encountered file will be assumed to contain the `.xml` output of a CAEVO parse, with  each sentence represented by an `entry` element, where the text is stored as `entry.sentence.text` and the parse tree is stored as `entry.parse.text`.
 * `--out_dir`: Target directory for resulting `.txt` files containing filtered parse trees. Will be created if nonexistent.
 

`process_tree.py --tree_path 'TREE_DIR/*' --out_dir EXAMPLE_DIR/`

 * `--tree_path`: Will be glob'd, every encountered file will be assumed to contain syntactic parse trees, separated by empty lines.
 * `--out_dir`: Target directory for resulting `.json` files containing examples. Will be created if nonexistent.

`distant.py`:

`examples.py`:
