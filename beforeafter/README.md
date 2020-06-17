# BeforeAfter Example Parsing

1. Generate syntactic parse tree and output to `TREE_DIR/<filename>_<rel>.tree/txt`. Two methods:
  
    1. Directly from Gigaword. 
    
        1. Generate candidate sentences: 
        
            `python filter_gz.py --files 'PATH/TO/gigaword_eng_5/data/SOURCE/*.gz', --out_dir FILTERED_DIR/`
        
        2. Generate syntactic parse tree using Stanford CoreNLP:
        
            `./parse_tree.sh /ABS/PATH/TO/STANFORD_NLP/ /ABS/PATH/TO/FILTERED_DIR/* /ABS/PATH/TO/TREE_DIR/`

    2. Extract from CAEVO parse.

        `python filter_xml.py --files 'PATH/TO/*.xml' --out_dir TREE_DIR/`

2. Process parse tree and extract event-event relation. Output to `EXAMPLE_DIR/<filename>_<rel>.json`.

    `python process_tree.py --tree_path 'TREE_DIR/*' --out_dir EXAMPLE_DIR/`


## Files

`python filter_gz.py --files 'PATH/TO/gigaword_eng_5/data/SOURCE/*.gz', --out_dir FILTERED_DIR/'`

    Filters gigaword .gz files to sentences containing before/after mentions

 * `--files`: Will be glob'd, every encountered file will be assumed to be a .gz file, which contains HTML when unzipped, where each sentence is a `<p>` element.
 * `--out_dir`: Target directory for resulting `.txt` files containing filtered sentences. Will be created if nonexistent.

`./parse_tree.sh /ABS/PATH/TO/STANFORD_NLP/ /ABS/PATH/TO/FILTERED_DIR/* /ABS/PATH/TO/TREE_DIR/`

    Generate .tree files from txt files in FILTERED_DIR using Stanford CoreNLP and outputs to TREE_DIR/


`python filter_xml.py --files 'PATH/TO/*.xml' --out_dir TREE_DIR/`

    Filters parse trees in XML to ones containing before/after mentions.

 * `--files`: Will be glob'd, every encountered file will be assumed to contain the `.xml` output of a CAEVO parse, with  each sentence represented by an `entry` element, where the text is stored as `entry.sentence.text` and the parse tree is stored as `entry.parse.text`.
 * `--out_dir`: Target directory for resulting `.tree` files containing filtered parse trees. Will be created if nonexistent.
 

`python process_tree.py --tree_path 'TREE_DIR/*' --out_dir EXAMPLE_DIR/`

 * `--tree_path`: Will be glob'd, every encountered file will be assumed to contain syntactic parse trees, separated by empty lines.
 * `--out_dir`: Target directory for resulting `.json` files containing examples. Will be created if nonexistent.

`distant.py`:

`examples.py`:
