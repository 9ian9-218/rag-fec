# A Performance Comparison of Polar Codes and Reed-Muller Codes

Erdal Arıkan, Senior Member, IEEE

Abstract—Polar coding is a code construction method that can be used to construct capacity-achieving codes for binary-input channels with certain symmetries. Polar coding may be considered as a generalization of Reed-Muller (RM) coding. Here, we demonstrate the performance advantages of polar codes over RM codes under belief-propagation decoding.

Index Terms—Polar codes, Reed-Muller (RM) codes, channel coding, forward error correction.

# I. INTRODUCTION

POLAR coding is a code construction method that can achieve the capacity of symmetric binary-input discrete memoryless channels such as the binary symmetric channel (BSC) and binary erasure channel (BEC). This technique was introduced and theoretically analyzed in [1]. The aim of this letter is to give some experimental results to show the practical utility of polar coding. Specifically, we demonstrate how to use the polar coding idea to improve the performance of Reed Muller (RM) codes [2], [3] without increasing their encoding and decoding complexity.

# II. RM CODES

Let $G_{RM}(n,n)$ denote the generator matrix of an nth order RM code of block-length $N = 2^n$ . Using the well-known Plotkin construction for RM codes, we may take

$$
G _ {R M} (n, n) = F ^ {\otimes n} \tag {1}
$$

where $F = \begin{bmatrix} 1 & 0 \\ 1 & 1 \end{bmatrix}$ and $F^{\otimes n}$ denotes the nth tensor power of F. The rth order RM code $\mathrm{RM}(r, n)$ can then be defined as the linear code with generator matrix $G_{RM}(r, n)$ which is obtained by taking the rows of $G_{RM}(n, n)$ with Hamming weights $\geq 2^{n-r}$ . For example, $G_{RM}(3, 3)$ is given by

$$
G _ {R M} (3, 3) = \left[ \begin{array}{c c c c c c c c} 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\ 1 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\ 1 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\ 1 & 1 & 1 & 1 & 0 & 0 & 0 & 0 \\ 1 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\ 1 & 1 & 0 & 0 & 1 & 1 & 0 & 0 \\ 1 & 0 & 1 & 0 & 1 & 0 & 1 & 0 \\ 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 \end{array} \right] \tag {2}
$$

Manuscript received January 4, 2008. The associate editor coordinating the review of this letter and approving it for publication was A. Haimovich. This work was supported in part by The Scientific and Technological Research Council of Turkey (TÜBİTAK) under project no 107E216, and in part by the EC under FP7 Network of Excellence project NEWCOM++.

E. Arıkan is with the Department of Electrical-Electronics Engineering, Bilkent University, Ankara, 06800, Turkey (e-mail: arıkan@ee.bilkent.edu.tr).

Digital Object Identifier 10.1109/LCOMM.2008.080017.

and RM(1,3) is the code with generator matrix

$$
G _ {R M} (1, 3) = \left[ \begin{array}{c c c c c c c c} 1 & 1 & 1 & 1 & 0 & 0 & 0 & 0 \\ 1 & 1 & 0 & 0 & 1 & 1 & 0 & 0 \\ 1 & 0 & 1 & 0 & 1 & 0 & 1 & 0 \\ 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 \end{array} \right] \tag {3}
$$

# III. POLAR CODES

For any $N = 2^{n}$ , $n \geq 1$ , and $1 \leq K \leq N$ , an $(N, K)$ polar code is a block code whose generator matrix $G_{P}(N, K)$ is a $K \times N$ submatrix of $F^{\otimes n}$ constructed in accordance with the following procedure. First, we compute the vector $z_{N} = (z_{N,1}, \ldots, z_{N,N})$ through the recursion

$$
z _ {2 k, j} = \left\{ \begin{array}{l l} 2 z _ {k, j} - z _ {k, j} ^ {2} & \text { for   } 1 \leq j \leq k \\ z _ {k, j - k} ^ {2} & \text { for   } k + 1 \leq j \leq 2 k \end{array} \right. \tag {4}
$$

for $k = 1,2,2^2,\ldots ,2^{n - 1}$ , starting with $z_{1,1} = 1 / 2$ . Next, we form a permutation $\pi_N = (i_1,\dots ,i_N)$ of the set $(1,\dots ,N)$ so that, for any $1\leq j < k\leq N$ , the inequality $z_{N,i_j}\leq z_{N,i_k}$ is true. The generator matrix $G_{P}(N,K)$ for an $(N,K)$ polar code is defined as the submatrix of $F^{\otimes n}$ consisting of rows with indices $i_1,\ldots ,i_K$ . It is easy to see that the computational complexity of this code construction method is $\mathcal{O}(N\log N)$ .

As an example, consider the case n = 3. Then, we have $z_{8} = (0.996, 0.684, 0.809, 0.121, 0.879, 0.191, 0.316, 0.004)$ , which gives $\pi_{8} = (8, 4, 6, 7, 2, 3, 5, 1)$ . Thus, an $(N, K) = (8, 5)$ polar code has the generator matrix

$$
G _ {P} (8, 5) = \left[ \begin{array}{c c c c c c c c} 1 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\ 1 & 1 & 1 & 1 & 0 & 0 & 0 & 0 \\ 1 & 1 & 0 & 0 & 1 & 1 & 0 & 0 \\ 1 & 0 & 1 & 0 & 1 & 0 & 1 & 0 \\ 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 \end{array} \right] \tag {5}
$$

Since the ranking of the rows of $F^{\otimes3}$ by $\pi_{8}$ happens to coincide with the ranking of them by their Hamming weights, there is no difference between polar coding and RM coding for n = 3. The same holds also for n = 4. However, for n = 5, we have $\pi_{32} = (32, 16, 24, 28, 30, 31, 8, 12, 20, 14, 22, 26, 15, 23, 27, 4, 29, 6, 10, 7, 18, 11, 19, 13, 21, 2, 25, 3, 5, 9, 17, 1)$ , while the weights of the corresponding rows are (32, 16, 16, 16, 16, 16, 8, 8, 8, 8, 8, 8, 8, 8, 4, 8, 4, 4, 4, 4, 4, 4, 4, 2, 4, 2, 2, 2, 2, 1). Thus, the polar code with parameter $(N, K) = (32, 16)$ differs from the RM code with the same parameter in that it employs a weight-4 row of $F^{\otimes5}$ and leaves out a weight-8 row. The non-equivalence of RM codes and polar codes becomes commonplace as the code order n increases and significant performance advantages begin to emerge in favor of polar codes, as we will show in the next section by experimental results.

As Forney [4] showed, RM codes can be regarded as codes on graphs, and, hence, decoded by BP decoders. Since polar

codes are subcodes of the full RM(n,n) codes, they, too, can be decoded by BP decoders. In this paper, we consider solely BP decoding of RM and polar codes. Using Forney's factor graph representation for RM codes, it is easy to see that the encoding and BP-decoding complexities of RM and polar codes are both $\mathcal{O}(N\log N)$ where N is the code block-length.

# IV. EXPERIMENTAL RESULTS

First, we give a result, as presented in Table I, for the case $(N, K) = (32, 16)$ , which is the smallest instance for which RM and polar codes differ. This table gives the bit error rate (BER) performance of $(32, 16)$ RM and polar codes over BECs with various erasure probabilities. (The final row of the table corresponds to a different type of polar coding scheme that will be discussed in the next section.) Each entry in this table was obtained by simulating the transmission of 10,000 codewords. To minimize statistical fluctuations, we used the same set of 10,000 channel realizations for all three entries in each column. Maximum number of iterations in BP decoding was set to 60, allowing each node in the code's factor graph to be visited up to 60 times.

TABLE I
BER FOR $(N, K) = (32, 16)$ CODES ON A BEC. 

<table><tr><td></td><td colspan="5">Erasure probability</td></tr><tr><td>Type of Code</td><td>0.05</td><td>0.15</td><td>0.25</td><td>0.35</td><td>0.45</td></tr><tr><td>RM(2,5)</td><td>0.00000</td><td>0.00039</td><td>0.01169</td><td>0.06525</td><td>0.24507</td></tr><tr><td>Polar</td><td>0.00001</td><td>0.00056</td><td>0.00702</td><td>0.05005</td><td>0.16722</td></tr><tr><td>Adaptive polar</td><td>0.00000</td><td>0.00039</td><td>0.00702</td><td>0.05005</td><td>0.16722</td></tr></table>

Table I does not provide sufficient evidence to conclude that polar codes are better than RM codes. However, as we consider codes with larger block-lengths, it does not take too long to obtain such evidence. To that end, we will consider codes of length N = 256 with dimensions K as given in Table II. Here, the code dimensions are chosen to correspond to those of RM(r,8) codes by setting $K = \sum_{i=0}^{r} \binom{8}{i}$ .

TABLE II
DIMENSIONS OF CODES USED IN SIMULATIONS. 

<table><tr><td>r</td><td>K</td><td>Rate</td></tr><tr><td>1</td><td>9</td><td>0.04</td></tr><tr><td>2</td><td>37</td><td>0.14</td></tr><tr><td>3</td><td>93</td><td>0.36</td></tr><tr><td>4</td><td>163</td><td>0.64</td></tr><tr><td>5</td><td>219</td><td>0.86</td></tr><tr><td>6</td><td>247</td><td>0.96</td></tr></table>

In Table III we show simulation results for RM codes of length N = 256 over a BEC. Results for competing polar codes are presented in Table IV. Comparison of the two sets of results shows that polar codes have a clear performance advantage. In these tables, and the ones that follow, we obtained each entry by simulating the transmission of 1000 codewords, with the maximum number of iterations in the BP decoder equal to 60. Blank entries in tables correspond to cases where code rate is above channel capacity.

Performance results for RM and polar codes on a BSC are shown in Tables VI and VII, respectively. Performances under inputs $\pm1$ ) over an additive Gaussian noise channel (AGNC) are shown in Tables VIII and IX. Inspection of these tables again shows a distinct performance advantage for polar codes over RM codes. We have observed experimentally that such advantages become even more pronounced as the code block-length is increased.

# V. CHANNEL-SPECIFIC CODE CONSTRUCTION

One important property of the code construction rule (4) is that it is channel-independent; we use the same rule for constructing codes for any binary-input channel. Results in [1] show that if one uses polar code construction rules that are tailored to the specific channel on which the code will be used, one may expect better performance at the expense of more complexity in code construction (but not in encoding and decoding of the resulting code). In fact, it is proved in [1] that such channel-specific rules can achieve channel capacity.

The channel-specific polar code construction rule given in [1] has a simple form for the class of BECs. For a BEC with erasure probability $\epsilon$ , the rule is the same as above except the recursion in (4) begins with $z_{1,1} = \epsilon$ . Thus, the code construction method that we considered above is actually tailored for a BEC with erasure rate 1/2, but it happens to work well for other channels as well. This is significant in that it shows the robustness of the rule against uncertainties and variations in channel parameters.

If we construct (32,16) polar codes tailored to specific BECs, we obtain the performance given in the last row of Table I, which matches the best performance available by either the RM code or the fixed polar code for the same scenario. Advantages of channel-tailored polar coding is illustrated further by Table V, which gives performance results for channel-adapted polar codes of length N = 256. Comparison of Tables V and IV shows that significant BER improvements are possible by channel-specific constructions.

Unfortunately, the exact code construction rule for arbitrary binary-input channels is too complicated to be given here. We will conclude by suggesting a heuristic method instead: given an arbitrary binary-input channel with capacity C bits, use the polar code that is matched to the BEC with erasure rate $\epsilon = 1 - C$ , i.e., the BEC that has same capacity as the given channel. This rule has yielded good results in experiments.

# REFERENCES

[1] E. Arikan, “Channel polarization: a method for constructing capacity-achieving codes for symmetric binary-input memoryless channels,” submitted for publication, Oct. 2007.   
[2] D. E. Muller, “Application of boolean algebra to switching circuit design and to error correction,” IRE Trans. Electron. Computers, vol. EC-3, pp. 6–12, Sept. 1954.   
[3] I. Reed, “A class of multiple-error-correcting codes and the decoding scheme,” IRE Trans. Inform. Theory, vol. 4, pp. 39–44, Sept. 1954.   
[4] G. D. Forney Jr., “Codes on graphs: normal realizations,” IEEE Trans. Inform. Theory, vol. IT-47, pp. 520–548, Feb. 2001.

TABLE III
RM CODE OVER BEC. N = 256, TRIALS = 1000, MAX. ITER. = 60. 

<table><tr><td></td><td colspan="9">Erasure probability</td></tr><tr><td>Rate</td><td>0.10</td><td>0.20</td><td>0.30</td><td>0.40</td><td>0.50</td><td>0.60</td><td>0.70</td><td>0.80</td><td>0.90</td></tr><tr><td>0.04</td><td>0.00000</td><td>0.00000</td><td>0.00000</td><td>0.00000</td><td>0.00000</td><td>0.00000</td><td>0.00000</td><td>0.00000</td><td>0.07200</td></tr><tr><td>0.14</td><td>0.00000</td><td>0.00000</td><td>0.00000</td><td>0.00000</td><td>0.00000</td><td>0.01195</td><td>0.24395</td><td>0.47465</td><td>-</td></tr><tr><td>0.36</td><td>0.00000</td><td>0.00000</td><td>0.00933</td><td>0.21003</td><td>0.45208</td><td>0.48368</td><td>-</td><td>-</td><td>-</td></tr><tr><td>0.64</td><td>0.00083</td><td>0.23588</td><td>0.45156</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>0.86</td><td>0.30484</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr></table>

TABLE IV
POLAR CODE OVER BEC. N = 256, TRIALS = 1000, MAX. ITER. = 60. 

<table><tr><td></td><td colspan="9">Erasure probability</td></tr><tr><td>Rate</td><td>0.10</td><td>0.20</td><td>0.30</td><td>0.40</td><td>0.50</td><td>0.60</td><td>0.70</td><td>0.80</td><td>0.90</td></tr><tr><td>0.04</td><td>0.00000</td><td>0.00000</td><td>0.00000</td><td>0.00000</td><td>0.00000</td><td>0.00000</td><td>0.00000</td><td>0.00000</td><td>0.01378</td></tr><tr><td>0.14</td><td>0.00000</td><td>0.00000</td><td>0.00000</td><td>0.00000</td><td>0.00000</td><td>0.00000</td><td>0.00308</td><td>0.20427</td><td>-</td></tr><tr><td>0.36</td><td>0.00000</td><td>0.00000</td><td>0.00000</td><td>0.00011</td><td>0.02087</td><td>0.29740</td><td>-</td><td>-</td><td>-</td></tr><tr><td>0.64</td><td>0.00000</td><td>0.00363</td><td>0.16155</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>0.86</td><td>0.10419</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr></table>

TABLE V
CHANNEL-SPECIFIC POLAR CODE OVER BEC. N = 256, TRIALS = 1000, MAX. ITER. = 60. 

<table><tr><td></td><td colspan="9">Erasure probability</td></tr><tr><td>Rate</td><td>0.10</td><td>0.20</td><td>0.30</td><td>0.40</td><td>0.50</td><td>0.60</td><td>0.70</td><td>0.80</td><td>0.90</td></tr><tr><td>0.04</td><td>0.00000</td><td>0.00000</td><td>0.00000</td><td>0.00000</td><td>0.00000</td><td>0.00000</td><td>0.00000</td><td>0.00000</td><td>0.00689</td></tr><tr><td>0.14</td><td>0.00000</td><td>0.00000</td><td>0.00000</td><td>0.00000</td><td>0.00000</td><td>0.00000</td><td>0.00130</td><td>0.11181</td><td>-</td></tr><tr><td>0.36</td><td>0.00000</td><td>0.00000</td><td>0.00000</td><td>0.00005</td><td>0.02087</td><td>0.29740</td><td>-</td><td>-</td><td>-</td></tr><tr><td>0.64</td><td>0.00000</td><td>0.00339</td><td>0.16155</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>0.86</td><td>0.08167</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr></table>

TABLE VI
RM CODE OVER BSC. N = 256, TRIALS = 1000, MAX. ITER. = 60. 

<table><tr><td></td><td colspan="11">Error probability</td></tr><tr><td>Rate</td><td>0.05</td><td>0.10</td><td>0.15</td><td>0.20</td><td>0.25</td><td>0.30</td><td>0.35</td><td>0.36</td><td>0.37</td><td>0.38</td><td>0.39</td></tr><tr><td>0.04</td><td>0.00000</td><td>0.00000</td><td>0.00000</td><td>0.00000</td><td>0.00000</td><td>0.01111</td><td>0.10756</td><td>0.14389</td><td>0.21100</td><td>0.25778</td><td>0.30400</td></tr><tr><td>0.14</td><td>0.00000</td><td>0.00000</td><td>0.00584</td><td>0.12159</td><td>0.34368</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>0.36</td><td>0.00086</td><td>0.12829</td><td>0.39794</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr></table>

TABLE VII
POLAR CODE OVER BSC. N = 256, TRIALS = 1000, MAX. ITER. = 60. 

<table><tr><td></td><td colspan="11">Error probability</td></tr><tr><td>Rate</td><td>0.05</td><td>0.10</td><td>0.15</td><td>0.20</td><td>0.25</td><td>0.30</td><td>0.35</td><td>0.36</td><td>0.37</td><td>0.38</td><td>0.39</td></tr><tr><td>0.04</td><td>0.00000</td><td>0.00000</td><td>0.00000</td><td>0.00000</td><td>0.00000</td><td>0.00344</td><td>0.04667</td><td>0.07844</td><td>0.11889</td><td>0.17722</td><td>0.21378</td></tr><tr><td>0.14</td><td>0.00000</td><td>0.00000</td><td>0.00000</td><td>0.01230</td><td>0.14297</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>0.36</td><td>0.00000</td><td>0.02418</td><td>0.21481</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr></table>

TABLE VIII
RM CODE USING ANTIPODAL SIGNALING OVER AGNC. N = 256, TRIALS = 1000, MAX. ITER. = 60. 

<table><tr><td></td><td colspan="11">Signal-to-noise ratio (dB)</td></tr><tr><td>Rate</td><td>-10.00</td><td>-8.00</td><td>-6.00</td><td>-4.00</td><td>-2.00</td><td>0.00</td><td>2.00</td><td>4.00</td><td>6.00</td><td>8.00</td><td>10.00</td></tr><tr><td>0.04</td><td>0.09300</td><td>0.02400</td><td>0.00089</td><td>0.00000</td><td>0.00000</td><td>0.00000</td><td>0.00000</td><td>0.00000</td><td>0.00000</td><td>0.00000</td><td>0.00000</td></tr><tr><td>0.14</td><td>-</td><td>-</td><td>0.42068</td><td>0.23214</td><td>0.02949</td><td>0.00014</td><td>0.00000</td><td>0.00000</td><td>0.00000</td><td>0.00000</td><td>0.00000</td></tr><tr><td>0.36</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.30597</td><td>0.02986</td><td>0.00000</td><td>0.00000</td><td>0.00000</td><td>0.00000</td></tr><tr><td>0.64</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.40810</td><td>0.10399</td><td>0.00109</td><td>0.00000</td><td>0.00000</td></tr><tr><td>0.86</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.12259</td><td>0.00051</td><td>0.00000</td></tr><tr><td>0.96</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.05732</td><td>0.00044</td></tr></table>

TABLE IX
POLAR CODE USING ANTIPODAL SIGNALING OVER AGNC. N = 256, TRIALS = 1000, MAX. ITER. = 60. 

<table><tr><td></td><td colspan="11">Signal-to-noise ratio (dB)</td></tr><tr><td>Rate</td><td>-10.00</td><td>-8.00</td><td>-6.00</td><td>-4.00</td><td>-2.00</td><td>0.00</td><td>2.00</td><td>4.00</td><td>6.00</td><td>8.00</td><td>10.00</td></tr><tr><td>0.04</td><td>0.04844</td><td>0.00400</td><td>0.00000</td><td>0.00000</td><td>0.00000</td><td>0.00000</td><td>0.00000</td><td>0.00000</td><td>0.00000</td><td>0.00000</td><td>0.00000</td></tr><tr><td>0.14</td><td>-</td><td>-</td><td>0.23784</td><td>0.03135</td><td>0.00038</td><td>0.00000</td><td>0.00000</td><td>0.00000</td><td>0.00000</td><td>0.00000</td><td>0.00000</td></tr><tr><td>0.36</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.04568</td><td>0.00088</td><td>0.00000</td><td>0.00000</td><td>0.00000</td><td>0.00000</td></tr><tr><td>0.64</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.25902</td><td>0.00787</td><td>0.00000</td><td>0.00000</td><td>0.00000</td></tr><tr><td>0.86</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.04048</td><td>0.00000</td><td>0.00000</td></tr><tr><td>0.96</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.05671</td><td>0.00068</td></tr></table>