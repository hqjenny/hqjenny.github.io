## Configuration Specs 
### 1. **Problem**  
```
R - kernel width
S - kernel height
P - output width
Q - output height
K - output channels
C - input channels  
N - batch size
```
### 2. **Arhictecture Storage**
The memory width is `word_bits x block_size x cluster_size`, height is `entries / block_size`
```
instance - number of memory instances
word-bits - bits per word
entries - number of word-bits entries per memory instance
size - total memory size in KB (one can specify either entries or size for the total memory size) 
block-size - number of total parallel ports 
cluster-size - number of reduction network elements
network-word-bits - bits per output word (accumulated output needs to be requantized to this width)
meshX - X dimension size on a mesh 
```
### 3. **Map Space**
For memory at each level, one can specify the *datatype*, *spatial*, *utilization*. 
Ordering of loops within a tiling-level determines the sequence in which sub-tiles will be delivered from that level to an inner level during execution. 

#### a. datatype
```
        {   
            target = "FakeDRAM";
            type = "datatype";
            keep =
            [   
                "Weights",
                "Inputs",
            ];  
            bypass =
            [   
                "Outputs"
            ];  
        },  
```
#### b. spatial 

`factors` specifies the tiling factor for each dimension. Unspecified factor (or zero) gives Timeloop full flexibility to determine an optimal value.
`permutation` specifies the loop ordering within a tiling level. (also specifies the sequence of tiling factor search and loop interchange?)

```
        # Spatial
        {
            target = "AccumulationBuffer";
            type = "spatial";
            factors = "P1 Q1 R1 S1 C64 K1 N1";
            permutation = "KQRSPNC";
        },
        {
            target = "WeightInputBuffer";
            type = "spatial";
            factors = "P1 Q1 R1 S1 C1 K16 N1";
            permutation = "KCQRSPN";
        },        
```
#### c.temperal 
`factors` specifies the tiling factor for each dimension. Unspecified factor (or zero) gives Timeloop full flexibility to determine an optimal value. In the following example, P and Q are the temperal tiling that can be any value within the tiling level. For each register, it can only store one value (filter width, filter height, channel input, channel output = 1).
```
        # Temporal
        {   
            target = "Registers";
            type = "temporal";
            factors = "R1 S1 C1 K1 N1"; # P Q free 
            permutation = "PQRSCKN";
        }, 
```
#### d. utilization
This specifies in what condition, the level of storage can be bypassed.

### 4. **Output**
Registers buffers 4096 inputs at once, but reload only 5760 times.

It stores 
```
==========================================
Registers
    Inputs tile: 5760
------------------------------------------
                for Q in [0,480)
                  for P in [0,12)
```
Registers only buffer 1 weight. 
```
==========================================
Registers
   Weights tile: 1
------------------------------------------
                for Q in [0,480)
                  for P in [0,48)
```

In the output file, typically the tile size is the dot product of inner loop bounds. 

- Weight tile = NRSKC
- Input tile = N(P+2)(Q+2)C
- Output = NPQK

Unless specified as spatial, the tiles are accessed temporally. 
Note that spatial paritioning of C results in a reduction tree with inputs from inner loop nest and outputs to the current loop nest. One can refer to `Scalar updates` in the output STATs to see how many times each tile is reloaded. 

## NVDLA Architecture Overview
This document explains the detailed archecture of [NVDLA](http://nvdla.org/hw/v1/hwarch.html)


