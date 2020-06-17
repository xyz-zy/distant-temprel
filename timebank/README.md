# MATRES/TimeBank Example Parsing

Minimally, to parse the MATRES examples, the following directory structure must be in place:

```
.
├── MATRES (submodule)
│   ├── aquaint.txt
│   ├── platinum.txt 
│   └── timebank.txt
├── TBAQ-cleaned
│   ├── AQUAINT
│   │   └── (.tml)
│   └── TimeBank
│       └── (.tml)
└── te3-platinum
    └── (.tml)
```

To do this:

1. Clone the `MATRES` submodule, if needed.

2. Download and unzip [TBAQ-cleaned](http://alt.qcri.org/semeval2015/task5/data/uploads/tbaq-2013-03.zip).

3. Download and unzip [te3-platinum](http://alt.qcri.org/semeval2015/task5/data/uploads/te3-platinumstandard.tar.gz)

  In the platinum docs, rename `nyt_20130321_sarkozy.tml` to `nyt_20130321_sarcozy.tml`, to account for a mismatch of doc names in MATRES annotations.

