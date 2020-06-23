# UDS-T Example Parsing

Contains code to parse UDS-T data into MATRES-stype classification examples. Relations are determined by relative start dates.

1. Download and unzip the [UDS-T annotations](http://decomp.io/projects/time/UDS_T_v1.0.zip).

2. Download and unzip releases [1.2](https://github.com/UniversalDependencies/UD_English-EWT/releases/tag/r1.2) and [2.5](https://github.com/UniversalDependencies/UD_English-EWT/releases/tag/r2.5) of the Universal Dependencies dataset. 

3. Generate parsed UD output JSON files.

    ```
    mkdir UD_English-r1.2_parsed/
    python get_ud_data.py --output UD_English-r1.2_parsed/en-ud-train.json --output_type json --ud_version 1.2 --input UD_English-EWT-r1.2/en-ud-train.conllu --split_by_25 --input_25 UD_English-EWT-r2.5/en_ewt-ud-train.conllu
    python get_ud_data.py --output UD_English-r1.2_parsed/en-ud-dev.json --output_type json --ud_version 1.2 --input UD_English-EWT-r1.2/en-ud-dev.conllu --split_by_25 --input_25 UD_English-EWT-r2.5/en_ewt-ud-dev.conllu
    python get_ud_data.py --output UD_English-r1.2_parsed/en-ud-test.json --output_type json --ud_version 1.2 --input UD_English-EWT-r1.2/en-ud-test.conllu --split_by_25 --input_25 UD_English-EWT-r2.5/en_ewt-ud-test.conllu
    ```

4. Generate train examples.

    ```
    mkdir all_annotations/
    python3 get_UDST_data.py --input UDS_T_v1.0/time_eng_ud_v1.2_2015_10_30.tsv --UD_inputs 'UD_English-r1.2_parsed/en-ud-*.json' --output_dir all_annotations/ --action json --src en-ud-train
    ```
5. Generate dev and test examples, leaving all annotations (3 per example).

    ```
    python3 get_UDST_data.py --input UDS_T_v1.0/time_eng_ud_v1.2_2015_10_30.tsv --UD_inputs 'UD_English-r1.2_parsed/en-ud-*.json' --output_dir all_annotations/ --action json --src en-ud-dev --preserve_all_anns
    python3 get_UDST_data.py --input UDS_T_v1.0/time_eng_ud_v1.2_2015_10_30.tsv --UD_inputs 'UD_English-r1.2_parsed/en-ud-*.json' --output_dir all_annotations/ --action json --src en-ud-test --preserve_all_anns
    ```

6. Generate dev and test examples, where ties are first broken by confidence scores where possible and otherwise discarded:

    ```
    mkdir maj_conf_nt/
    python3 get_UDST_data.py --input UDS_T_v1.0/time_eng_ud_v1.2_2015_10_30.tsv --UD_inputs 'UD_English-r1.2_parsed/en-ud-*.json' --output_dir maj_conf_nt/ --action json --src en-ud-dev --rel_score maj --remove_ties
    python3 get_UDST_data.py --input UDS_T_v1.0/time_eng_ud_v1.2_2015_10_30.tsv --UD_inputs 'UD_English-r1.2_parsed/en-ud-*.json' --output_dir maj_conf_nt/ --action json --src en-ud-test --rel_score maj --remove_ties
    ```

## Example files

`all_annotations/` contains an example per pairwise annotation. For dev and test sets, there are multiple annotations per example.

`maj_conf_nt/` contains dev and test examples, with multple annotations resolved by majority vote. Ties are first broken by confidence score, and then discarded if they cannot be broken.

