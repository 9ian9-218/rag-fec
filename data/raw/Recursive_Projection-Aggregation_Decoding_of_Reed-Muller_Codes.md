# Recursive Projection-Aggregation Decoding of Reed-Muller Codes

Min Ye and Emmanuel Abbe

Abstract— We propose a new class of efficient decoding algorithms for Reed-Muller (RM) codes over binary-input memoryless channels. The algorithms are based on projecting the code on its cosets, recursively decoding the projected codes (which are lower-order RM codes), and aggregating the reconstructions (e.g., using majority votes). We further provide extensions of the algorithms using list-decoding. We run our algorithm for AWGN channels and Binary Symmetric Channels at the short code length (≤ 1024) regime for a wide range of code rates. Simulation results show that in both low code rate and high code rate regimes, the new algorithm outperforms the widely used decoder for polar codes (SCL+CRC) with the same parameters. The performance of the new algorithm for RM codes in those regimes is in fact close to that of the maximal likelihood decoder. Finally, the new decoder naturally allows for parallel implementations.

Index Terms— Reed-Muller codes, polar codes, RPA decoding, AWGN channels, binary symmetric channels.

## I. INTRODUCTION

R EED-MULLER (RM) codes are among the oldest fam-ilies of error-correcting codes [2]. The recent break- ilies of error-correcting codes [2]. Therecent breakthrough of polar codes [3] has brought the attention back to RM codes, due to the closeness of the two codes. RM codes have in particular the advantage of having a simple and universal code construction, and promising performances were demonstrated in several works [4], [5], with a scaling law conjectured to be comparable of that of random codes.

RM codes do not possess yet the generic analytical framework of polar codes (i.e., polarization theory). It was recently shown that RM codes achieve capacity on the Binary Erasure Channel (BEC) at constant rate [6], as well as for extremal rates for BEC and Binary Symmetric Channels (BSC) [7], but obtaining such results for a broader class of communication channels and rates remains open. Recent progress was made on these questions with a polarization approach to RM codes shown in [8]. See also [9] for a recent survey on RM codes.

Various decoding algorithms have been proposed for RM codes, starting with Reed algorithm [2], [10], and four important more recent line of works including automorphism group based decoding [11]–[13], recursive list-decoding [14]–[16], a new Berlekamp-Welch type of algorithm [17], [18], and a new algorithm utilizing minimum-weight parity checks [19]. In particular, [11], [14]–[18] give fairly powerful theoretical guarantees for efficient decoding of RM codes in specific regimes. However, there is not a thorough comparison between the performance of RM codes under these decoders and the performance of the widely used CRC-aided polar codes under the Successive Cancellation List (SCL) decoders [20].

In this paper, we propose a new class of decoding algorithms for Reed-Muller codes over any binary-input memoryless channels and compare its performance with polar codes. The new algorithms are based on recursive projections and aggregations of cosets decoding, exploiting the self-similarity of RM codes, and are extended with Chase list-decoding algorithms [21]. We run our new algorithms at the short code length (≤ ) regime for a wide range of code rates. 1024Simulation results show that the new algorithms improve on the widely used decoding algorithm for polar codes [20] in both low code rate and high code rate regimes. These are the type of regimes where polar codes are planned to enter the 5G standards [22] as well as relevant regimes for applications in the Internet of Things (IoT).

More specifically, we compare our new algorithm for RM codes with the Successive Cancellation List (SCL) decoder for CRC-aided polar codes [20], where we set the CRC size to take optimal values.1 For AWGN channels, our new algorithm has about . dB gain (more in some cases) over 0 5polar codes in various short code length (≤ ) and low code rate (≤ . ) regimes, and similar improvements are also 0 5obtained for BSC channels. Moreover, the performance of our new decoding algorithm is comparable to the best previously known algorithms for RM codes [16].

In the above regimes, the decoding error probability of our new algorithm is in fact shown to be close to that of the Maximal Likelihood decoder on RM codes. Some extensions and variants to potentially further improve the performance are also discussed, as well as possible extensions of the projection-aggregation algorithms to other families of codes.

In Section II, we give a high level description of the new type of algorithms. In Section III, we present decoding algorithm for BSC channels. In Section IV we generalize the algorithms to decode RM codes over any binary-input channel. Finally, in Section VI we present simulation results. In addition to the previously mentioned improvements over polar codes, we also empirically validate the improved scaling-law of RM codes over polar codes on BSC channels [23].

## II. A HIGH-LEVEL DESCRIPTION OFTHE NEW ALGORITHMS

We begin with some notation and background on RM codes. In this paper, we use ⊕ to denote sums over $\mathbb { F } _ { 2 } .$ . Let us consider the polynomial ring $\mathbb { F } _ { 2 } [ Z _ { 1 } , Z _ { 2 } , \ldots , Z _ { m } ]$ of m variables. Since $Z ^ { 2 } = Z$ in $\mathbb { F } _ { 2 }$ [, the following set of $2 ^ { m }$ monomials forms a =basis of $\mathbb { F } _ { 2 } [ Z _ { 1 } , Z _ { 2 } , \ldots , Z _ { m } ]$

$$
\{ \prod _ { i \in A } Z _ { i } : A \subseteq [ m ] \} , { \mathrm { ~ w h e r e ~ } } \prod _ { i \in \emptyset } Z _ { i } : = 1 .
$$

Next we associate every subset $A \subseteq [ m ]$ with a row vector $\nu _ { m } ( A )$ of length $2 ^ { m }$ , whose components are indexed by a ( )binary vector $z = ( z _ { 1 } , z _ { 2 } , \ldots , z _ { m } ) \in \{ 0 , 1 \} ^ { m }$ . The vector $\nu _ { m } ( A )$ = (is defined as follows

$$
\nu _ { m } ( A , z ) = \prod _ { i \in A } z _ { i } ,\tag{1}
$$

where $\nu _ { m } ( A , z )$ is the component of $\nu _ { m } ( A )$ indexed by z, $\mathrm { i } . \mathrm { e } . , \nu _ { m } ( A , z )$ is the evaluation of the monomial $\Pi _ { i \in A } Z _ { i }$ at z. For $0 \leq r \leq m$ , the set of vectors

$$
\{ \pmb { \nu } _ { m } ( A ) : A \subseteq [ m ] , | A | \leq r \}
$$

forms a basis of the r-th order Reed-Muller code $\mathcal { R M } ( m , r )$ of length $n : = 2 ^ { m }$ and dimension $\textstyle \sum _ { i = 0 } ^ { r } { \binom { m } { i } }$

:= 2Definition 1: The r-th order Reed-Muller code $\mathcal { R M } ( m , r )$ code is defined as the following set of binary vectors

$$
\mathcal R \mathcal M ( m , r ) : = \Big \{ \sum _ { A \subseteq [ m ] , | A | \leq r } u ( A ) \pmb { \nu } _ { m } ( A ) : u ( A ) \in \{ 0 , 1 \}  \\  \qquad \quad \mathrm { ~ f o r ~ a l l ~ } A \subseteq [ m ] , | A | \leq r \Big \} .
$$

In other words, each vector $\nu _ { m } ( A )$ consists of all the evaluations of the monomial $\Pi _ { i \in A } Z _ { i }$ )at all the points in the vector space $\mathbb { E } : = \mathbb { F } _ { 2 } ^ { m }$ , and each codeword $c \in \mathcal { R M } ( m , r )$ := ( )corresponds to an m-variate polynomial with degree at most r. The coordinates of the codeword c are also indexed by the binary vectors $z \in \mathbb { E }$ , and we write $c = ( c ( z ) , z \in \mathbb { E } )$ . Let B be an s-dimensional subspace of E, where $s \leq r .$ ). The quotient space $\mathbb { E } / \mathbb { B }$ consists of all the cosets of B in E, where every coset T has form $T = z + \mathbb { B }$ for some $z \in \mathbb { E }$ . For a binary vector $y = ( y ( z ) , z \in \mathbb { E } )$ , we define its projection on the cosets of B as

$$
y _ { / \mathbb { B } } = \mathrm { P r o j } ( y , \mathbb { B } ) : = { \Big ( } y _ { / \mathbb { B } } ( T ) , T \in \mathbb { E } / \mathbb { B } { \Big ) } ,\tag{2}
$$

where $y _ { / \mathbb { B } } ( T ) : = \bigoplus _ { z \in T } y ( z )$ is the binary vector obtained by ( ) := ( )summing up all the coordinates of y in each coset $T \in \mathbb { E } / \mathbb { B }$ . Here the sum is over $\mathbb { F } _ { 2 }$ and the dimension of $y _ { / \mathbb { B } }$ is $n / | \mathbb { B } |$

In the next section, we will show that if c is a codeword of $\mathcal { R M } ( m , r )$ , then $c / \mathbb { B }$ is a codeword of $\mathcal { R M } ( m - s , r - s )$ ， where s is the dimension of B. Our new decoding algorithm makes use of the case $s = 1$ , namely, the one-dimensional =subspaces. More precisely, let $y = ( y ( z ) , z \in \mathbb { E } )$ be the output = ( (vector of transmitting a codeword of $\mathcal { R M } ( m , r )$ over some ( )BSC channel. Our decoding algorithm is defined in a recursive way: For every one-dimensional subspace B, we first obtain the projection $y _ { / \mathbb { B } } ,$ and then we use the decoding algorithm for $\mathcal { R M } ( m - 1 , r - 1 )$ to decode $y _ { / \mathbb { B } } ,$ , where the decoding result ( 1is denoted as $\hat { y } _ { / \mathbb { B } }$ 1). Since every one-dimensional subspace of ˆE consists of  and a non-zero element, there are $n - 1$ such 0 1subspaces in total. After the projection and recursive decoding steps, we obtain $n - 1$ decoding results $\hat { y } _ { / \mathbb { B } _ { 1 } } , \hat { y } _ { / \mathbb { B } _ { 2 } } , \dots , \hat { y } _ { / \mathbb { B } _ { n - 1 } } .$ 1 ˆ ˆ ˆNext we use a majority voting scheme to aggregate these decoding results together with y to obtain a new estimate $\hat { y }$ of the original codeword. Finally we update $y \operatorname { a s } { \hat { y } } .$ ˆ, and iterate the whole procedure for up to $N _ { \mathrm { m a x } }$ ˆrounds. Notice that if $y = \hat { y }$ (see line 6), then y is a fixed (stable) point of this algorithm and will remain unchanged for the next iterations. In this case we should exit the for loop on line 1 (see line 6–8). In practice we set the maximal number of iterations $N _ { \mathrm { m a x } } = \lceil m / 2 \rceil$ to = 2prevent the program from running into an infinite loop, and typically $\lceil m / 2 \rceil$ iterations are enough for the algorithm to con-2verge to a stable y. This high-level description is summarized in Fig. 1 and Algorithm 1. While this description focuses on the decoding algorithm over BSC, a natural extension of this algorithm bases on log-likelihood ratios (LLRs) allows us to decode RM codes over any binary-input memoryless channels, including the AWGN channel; see Section IV for details.

Algorithm 1 The RPA\_RM Decoding Function for BSC   
Input: The corrupted codeword $y \ = \ ( y ( z ) , z \ \in \ \mathbb { E } ) ;$ ; the   
= ( ( ) )parameters of the Reed-Muller code m and r; the maximal   
number of iterations $N _ { \mathrm { m a x } }$   
Output: The decoded codeword c   
1: for $j = 1 , 2 , \dots , N _ { \mathrm { m a x } }$ do   
2: $y _ { / \mathbb { B } _ { i } }  \mathrm { P r o j } ( y , \mathbb { B } _ { i } )$ for $i = 1 , 2 , \ldots , 2 ^ { m } - 1$   
3: $\hat { y } _ { / \mathbb { B } _ { i } } ~ \gets ~ \mathsf { R P A \_ R M } ( y _ { / \mathbb { B } _ { i } } , m - 1 , r - 1 , N _ { \operatorname* { m a x } } )$ for $i \ =$   
$1 , 2 , \ldots , 2 ^ { m } - 1$   
4: $\triangleright \operatorname { I f } r = 2 ,$ 1 then we use the Fast Hadamard Transform   
= 2to decode the first-order RM code [10]   
5: y ← Aggregat $\mathsf { . o n } ( y , \hat { y } _ { / \mathbb { B } _ { 1 } } , \hat { y } _ { / \mathbb { B } _ { 2 } } \ldots , \hat { y } _ { / \mathbb { B } _ { n - 1 } } )$   
6: ˆif $y = \hat { y }$ then   
7: = ˆbreak $\triangleright y = \hat { y }$ means that the algorithm already   
= ˆconverges to a fixed (stable) point   
8: end if   
9: $y \gets \hat { y }$   
10: end for   
11: $\hat { c } \gets \hat { y }$   
ˆ ˆ12: return c

## A. List Decoding Procedure [21]

Here we recap (a version of) the list decoding procedure proposed by Chase [21] that can further decrease the decoding error probability. Suppose that we have a unique decoding algorithm decodeC for some code C over some binary-input memoryless channel $W : \{ 0 , 1 \} \to \mathcal { W }$ . Without loss of generality, assume that decodeC is based on the LLR vector of the channel output, where the LLR of an output symbol $x \in \mathcal { W }$ is defined as

![](images/481b3a5e602045e9a3aeb7b2808f9ab19f14e8219c7ab2211c5fce4fe86d9b52.jpg)  
Fig. 1. Recursive Projection-Aggregation decoding algorithm for third order RM codes.

$$
\mathrm { L L R } ( x ) : = \ln { \Big ( } { \frac { W ( x | 0 ) } { W ( x | 1 ) } } { \Big ) } .\tag{3}
$$

Clearly, $\operatorname { i f } \mid \operatorname { L L R } ( x ) \mid$ is small, then x is a noisy symbol, and $\operatorname { i f } \mid \operatorname { L L R } ( x )$ LLR( )| is large, then $x$ is relatively noiseless.

LLR( )The list decoding procedure works as follows. Suppose that $y = ( y _ { 1 } , y _ { 2 } , \dotsc , y _ { n } )$ is the output vector when we send a code-= ( )word of C over the channel W . We first sort $| \operatorname { L L R } ( y _ { i } ) | , i \in [ n ]$ LLR( ) [ ]from small to large. Without loss of generality, let us assume that $| \operatorname { L L R } ( y _ { 1 } ) | , | \operatorname { L L R } ( y _ { 2 } ) | , | \operatorname { L L R } ( y _ { 3 } ) |$ are the three smallest LLR( ) LLR( ) LLR( )components in the LLR vector, meaning that $y _ { 1 } , y _ { 2 }$ and y3 are the three most noisy symbols in the channel outputs (we take three arbitrarily). Next we enumerate all the possible cases of the first three bits of the codeword $c = ( c _ { 1 } , c _ { 2 } , \ldots , c _ { n } ) \colon$ : The first three bits $( c _ { 1 } , c _ { 2 } , c _ { 3 } )$ = ( )can be any vector in F3 , so there ( )are cases in total, and for each case we change the value of $\mathrm { L L R } ( y _ { 1 } ) , \mathrm { L L R } ( y _ { 2 } ) , \mathrm { L L R } ( y _ { 3 } )$ according to the values of $c _ { 1 } , c _ { 2 } , c _ { 3 }$ ( ) LLR( ) LLR( ). More precisely, we set $\operatorname { L L R } ( y _ { i } ) ~ = ~ ( - 1 ) ^ { c _ { i } } L _ { \operatorname * { m a x } }$ for $i \ = \ 1 , 2 , 3$ , where $L _ { \mathrm { m a x } }$ LLR( ) = ( 1)is some large real number. = 1 2 3In practice, we can choose $L _ { \operatorname* { m a x } } : = \operatorname* { m a x } ( | \operatorname { L L R } ( y _ { i } ) | , i \in [ n ] )$ or $L _ { \operatorname* { m a x } } : = 2 \operatorname* { m a x } ( | \mathrm { L L R } ( y _ { i } ) | , i \in [ n ] )$ ( LLR( ) [ ]). For each of these := 2 max( LLR( ) [ ]) cases, we use decodeC to obtain a decoded codeword, and we denote them as $\hat { c } ^ { ( 1 ) } , \hat { c } ^ { ( 2 ) } , \dots , \hat { c } ^ { ( 8 ) }$ . Finally, we calculate the ˆposterior probability of $W ^ { n } ( y | \hat { c } ^ { ( i ) } ) , 1 \leq i \leq 8$ , and choose the largest one as the final decoding result, namely, we perform a maximal likelihood decoding among the  candidates in the list.

When we apply this list decoding procedure together with Algorithm 1 to decode RM codes, the decoding error probability is typically close to that of the Maximal Likelihood decoder.

## III. DECODING ALGORITHM FOR BSC

We begin with the definition of the quotient code. Then we show that the quotient code of an RM code is also an RM code.

Definition 2: Let $s \leq r \leq m$ be integers, and let B be an s-dimensional subspace of $\mathbb { E } : = \mathbb { F } _ { 2 } ^ { m }$ . We define the quotient code

$$
\mathcal { Q } ( m , r , \mathbb { B } ) : = \{ c _ { / \mathbb { B } } : c \in \mathcal { R } \mathcal { M } ( m , r ) \} .
$$

Lemma 1: Let $\textit { s } \leq \textit { r } \leq$ m be integers, and let B be an s-dimensional subspace of $\mathbb { E } : = \mathbb { F } _ { 2 } ^ { m }$ . The code $\mathcal { Q } ( m , r , \mathbb { B } )$ is the Reed-Muller code $\mathcal { R M } ( m - s , r - s )$

( )This lemma is an immediate corollary of Theorem 12 in [10, Chapter 13]. For the sake of completeness, we give a proof of this lemma in Appendix A.

Note that Reed’s algorithm [2] relies on the special case of $s = r$ in Lemma 1, and our new decoding algorithm makes =use of the case $s ~ = ~ 1$ in Lemma 1 (in addition to using = 1all subspaces and adding an iterative process). The RPA\_RM decoding function is already presented in the previous section. Here we fill in the only missing component, namely the Aggregation function; see Algorithm 2 below. Both $y _ { / \mathbb { B } _ { i } } =$ $( y _ { / \mathbb { B } _ { i } } ( T ) , T \in \mathbb { E } / \mathbb { B } )$ and $\hat { y } _ { / \mathbb { B } _ { i } } ~ = ~ ( \hat { y } _ { / \mathbb { B } _ { i } } ( T ) , T ~ \in ~ \mathbb { E } / \mathbb { B } )$ are ( ( ) )indexed by the cosets $T \in \mathbb { E } / \mathbb { B }$ = (ˆ ( ), and we use $[ z + \mathbb { B } ]$ )to denote the coset containing z (see line 3).

From line 3, we can see that the maximal possible value of changevote z for each $z \in \mathbb { E }$ is $n - 1$ . Therefore the ( )condition changevot $\begin{array} { r } { \mathbf { \tilde { \rho } } ( z ) > \frac { n - 1 } { 2 } } \end{array}$ 1on line can indeed be ( ) 4viewed as a majority vote. As discussed in Section $\mathrm { I I I - A } ,$ this algorithm can be viewed as one step of the power iteration method to find the eigenvector of a matrix built from the quotient code decoding.

In Algorithms 1–2, we write the pseudo codes in a mathematical fashion for the ease of understanding. In Appendix $\mathrm { C } ,$ we present another version of the RPA\_RM function in a program language fashion; see Algorithm 8.

Algorithm 2 The Aggregation Function for BSC   
Input: $y , \hat { y } _ { / \mathbb { B } _ { 1 } } , \hat { y } _ { / \mathbb { B } _ { 2 } } \ldots , \hat { y } _ { / \mathbb { B } _ { n - 1 } }$   
ˆOutput: y   
1: Initialize changevote $\mathbf { \varepsilon } ( z ) , z \in \{ 0 , 1 \} ^ { m } )$ as an all-zero   
(vector indexed by $z \in \{ 0 , 1 \} ^ { m }$   
2: $n \gets 2 ^ { m }$   
23: changevote $\begin{array} { r } { \mathbf { \varepsilon } ( z ) \gets \sum _ { i = 1 } ^ { n - 1 } \mathbb { 1 } \big [ \{ y _ { / \mathbb { B } _ { i } } ( [ z + \mathbb { B } _ { i } ] ) \neq \hat { y } _ { / \mathbb { B } _ { i } } ( [ z + } \end{array}$   
$\mathbb { B } _ { i } ] ) ]$ for each $z \in \{ 0 , 1 \} ^ { m }$   
4 $: y ( z )  y ( z ) \oplus \mathbb { 1 } [ \mathrm { c h a n g e v o t e } ( z ) > { \frac { n - 1 } { 2 } } ]$ for each $z \in$   
$\{ 0 , 1 \} ^ { m }$ ( ) ]- Here addition is over $\mathbb { F } _ { 2 }$   
0 15: return y

Proposition 1: The complexity of Algorithm 1 is $O ( n ^ { r } \log { n } )$ in sequential implementation and $O ( n ^ { 2 } )$ in parallel implementation with $O ( n ^ { r } )$ processors.

( )In Section VI-C, we further discuss options to reduce the computation time by using fewer subspaces in the projection step.

Proof: We prove by the induction on the order of the RM code. To establish the base case, observe that the complexity of decoding first-order RM codes using Fast Hadamard Transform (FHT) [10], [24] is $O ( n \log n )$ . Now we assume ( lthe proposition holds for decoding $( r \mathrm { ~ - ~ } 1 )$ -th order RM ( 1)codes and prove the inductive step. Clearly, the complexity of Algorithm 1 is determined by the complexity of the recursive decoding step on line 3. By induction hypothesis, the complexity of decoding each $y _ { / \mathbb { B } _ { i } }$ is $O ( n ^ { r - 1 } \log n )$ . Since (there are n −  one-dimensional subspaces $\mathbb { B } _ { 1 } , \mathbb { B } _ { 2 } , \ldots , \mathbb { B } _ { n - 1 } ,$ 1the complexity of Algorithm 1 is indeed $O ( n ^ { r } \log { n } )$

( log )In the next proposition, we show that whether Algorithm 1 outputs the correct codeword or not is independent of the transmitted codeword and only depends on the error pattern imposed by the BSC channel.

Proposition 2: Let $c \in \mathcal { R M } ( m , r )$ be a codeword of the RM code. Let $e = ( e ( z ) , z \in \mathbb { E } )$ ( )be the error vector imposed = ( ( ) )on c by the BSC channel, and the output vector of the BSC channel is $y = c + e$ . Denote the decoding result as $\hat { c } = \mathtt { R P A \_ R M } ( y , m , r , N _ { \operatorname* { m a x } } )$ . Then the indicator function of ˆ = (decoding error $\mathbb { I } [ \hat { c } \neq c ]$ is independent of the choice of c and [ˆ = ]only depends on the error vector e.

Notice that we use maximal likelihood decoder for first-order RM code, and the proposition can be proved by induction on the order of the RM code.2 This proposition is useful for simulations because we can simply transmit the all-zero codeword over the BSC channel to measure the decoding error probability.

## A. Spectral Interpretations of Algorithm 2

Algorithm 2 can be viewed as a one-step power iteration of a spectral algorithm. More precisely, observe that $\hat { y } _ { / \mathbb { B } _ { 1 } } , \hat { y } _ { / \mathbb { B } _ { 2 } } \ldots , \hat { y } _ { / \mathbb { B } _ { n - 1 } }$ contain the estimates of $c ( z ) \oplus c ( z ^ { \prime } )$ for all $z \neq z ^ { \prime } ,$ where $c = ( c ( z ) , z \in \mathbb { E } )$ is the transmitted (true) = = ( ( ) )codeword. {We denote the estimate of $c ( z ) \oplus c ( z ^ { \prime } )$ as $\hat { y } _ { z , z ^ { \prime } } .$ ( ) ( ) ˆSuppose for the moment that we want to find a vector $\hat { y } = ( \hat { y } ( z ) , z \in \mathbb { E } ) \in \{ 0 , 1 \} ^ { n }$ to agree with as many estimates ˆ = (ˆ( ) ) 0 1of these sums as possible, i.e., we want to find a vector $\hat { y }$ to maximize

$$
| \{ ( z , z ^ { \prime } ) : z \neq z ^ { \prime } , \hat { y } ( z ) \oplus \hat { y } ( z ^ { \prime } ) = \hat { y } _ { z , z ^ { \prime } } \} | .
$$

Notice that

$$
\begin{array} { r l } & { | \{ ( z , z ^ { \prime } ) : z \neq z ^ { \prime } , \hat { y } ( z ) \oplus \hat { y } ( z ^ { \prime } ) = \hat { y } _ { z , z ^ { \prime } } \} | } \\ & { + \left| \{ ( z , z ^ { \prime } ) : z \neq z ^ { \prime } , \hat { y } ( z ) \oplus \hat { y } ( z ^ { \prime } ) \neq \hat { y } _ { z , z ^ { \prime } } \} \right| = n ( n - 1 ) . } \end{array}
$$

Therefore,

$$
\begin{array} { r l } & { \displaystyle \sum _ { z \neq z ^ { \prime } } ( - 1 ) ^ { \hat { y } ( z ) + \hat { y } ( z ^ { \prime } ) + \hat { y } _ { z , z ^ { \prime } } } } \\ & { \displaystyle = 2 \left| \{ ( z , z ^ { \prime } ) : z \neq z ^ { \prime } , \hat { y } ( z ) \oplus \hat { y } ( z ^ { \prime } ) = \hat { y } _ { z , z ^ { \prime } } \} \right| - n ( n - 1 ) . } \end{array}
$$

Thus our task is equivalent to find

$$
\operatorname { a r g m a x } _ { \hat { y } \in \{ 0 , 1 \} ^ { n } } \sum _ { z \neq z ^ { \prime } } ( - 1 ) ^ { \hat { y } ( z ) + \hat { y } ( z ^ { \prime } ) + \hat { y } _ { z , z ^ { \prime } } } .\tag{4}
$$

Given a vector $\hat { y } \in \mathsf { ~ \Omega ~ } \{ 0 , 1 \} ^ { n }$ , we define another vector $\hat { u } \in \{ - 1 , 1 \} ^ { n }$ ˆby setting $\hat { u } ( z ) : = ( - 1 ) ^ { \hat { y } ( z ) }$ for all $z \in \mathbb { E } .$ ˆ 1 1 ˆ( ) := ( 1)In order to find the maximizing vector y in (4), it suffices to find

$$
\operatorname { a r g m a x } _ { \hat { u } \in \{ - 1 , 1 \} ^ { n } } \sum _ { z \neq z ^ { \prime } } ( - 1 ) ^ { \hat { y } _ { z , z ^ { \prime } } } \hat { u } ( z ) \hat { u } ( z ^ { \prime } ) .\tag{5}
$$

Now we build an n×n matrix A from $\{ \hat { y } _ { z , z ^ { \prime } } : z , z ^ { \prime } \in \mathbb { E } , z \neq z ^ { \prime } \}$ ˆ :as follows: The rows and columns of A are indexed by $z \in \mathbb { E } .$ and we set the entry

$$
\begin{array} { r } { A _ { z , z ^ { \prime } } : = \left\{ \begin{array} { l l } { ( - 1 ) ^ { \hat { y } _ { z , z ^ { \prime } } } } & { \mathrm { ~ i f ~ } z \neq z ^ { \prime } } \\ { \ \quad 0 } & { \mathrm { ~ i f ~ } z = z ^ { \prime } } \end{array} \right. , } \end{array}
$$

i.e., for $z \neq z ^ { \prime }$ we set $A _ { z , z ^ { \prime } } = 1 \mathrm { i f } \hat { y } _ { z , z ^ { \prime } } = 0 .$ , and $A _ { z , z ^ { \prime } } = - 1$ if $\hat { y } _ { z , z ^ { \prime } } = 1$ = = 1 ˆ = 0 = 1. Under this definition, the optimization problem (5) ˆ = 1becomes

$$
\begin{array} { r l } & { \operatorname { a r g m a x } _ { \hat { u } \in \{ - 1 , 1 \} ^ { n } } \displaystyle \sum _ { z \neq z ^ { \prime } } A _ { z , z ^ { \prime } } \hat { u } ( z ) \hat { u } ( z ^ { \prime } ) } \\ & { = \operatorname { a r g m a x } _ { \hat { u } \in \{ 1 , - 1 \} ^ { n } } \hat { u } ^ { T } A \hat { u } . } \end{array}\tag{6}
$$

It is well known that this combinatorial optimization problem is NP-hard. In practice, people usually use the following spectral relaxation to obtain approximate solution:

$$
\operatorname { a r g m a x } _ { \hat { u } \in \mathbb { R } ^ { n } , \| \hat { u } \| ^ { 2 } = n } \hat { u } ^ { T } A \hat { u } .
$$

It is well known that the solution to this relaxed optimization problem is the eigenvector corresponding to the largest eigenvalue of A. One way to find this eigenvector is to use the power iteration method: pick some vector v (e.g., at random), then $A ^ { t } v$ converges to this eigenvector when t is large enough.3 After rescaling $A ^ { t } v$ to make $\| A ^ { t } v \| ^ { 2 } = n .$ ， we obtain the maximizing vector $\tilde { u } ~ = ~ A ^ { t } v$ =in the relaxed ˜ =optimization problem. In order to obtain the solution to the original optimization problem in (6), we only need to look at the sign of each coordinate of u: If $\tilde { u } ( z ) > 0 .$ , then we set $\hat { u } ( z ) = 1$ , and if $\tilde { u } ( z ) < 0$ , then we set $\hat { u } ( z ) = - 1$ . In this ˆ( ) = 1 ˜( ) 0 ˆ( ) = 1way, we obtain the vector u that serves as our approximate ˆsolution to (6). To summarize, our approximate solution to (6) is $\hat { u } = \mathrm { s i g n } ( A ^ { t } v )$ , where v is some random vector and t is ˆ = sign( )some large enough integer.

Let us denote the output vector of Algorithm 2 as y, and we define another vector u as $\overline { { u } } ( z ) = ( - 1 ) ^ { \overline { { y } } ( z ) }$ for all $z \in \mathbb { E } ,$ ( ) = ( 1)For the original received vector y, we also define a vector u as $u ( z ) = ( - 1 ) ^ { y ( z ) }$ for all $z \in \mathbb { E } .$ The main observation in this subsection is that

$$
{ \overline { { u } } } = \operatorname { s i g n } ( A u ) ,\tag{7}
$$

i.e., the output of Algorithm 2 is in fact the same as a one-step power iteration of the spectral algorithm with the original received vector u playing the role of vector v above. It is also easy to see why (7) holds: According to $( 7 ) , \overline { { u } } ( z ) = 1$ $\begin{array} { r } { \mathrm { i f } \sum _ { z ^ { \prime } \ne z } \dot { ( - 1 ) } ^ { \hat { y } _ { z , z ^ { \prime } } \oplus y ( \bar { z ^ { \prime } } ) } > 0 } \end{array}$ and $\overline { { u } } ( z ) = - 1$ ( ) = 1otherwise. This is ( 1)equivalent to saying that $\overline { { y } } ( z ) = 0 \mathrm { i f } | \{ z ^ { \prime } : z ^ { \prime } \neq z , \hat { y } _ { z , z ^ { \prime } } \oplus y ( z ^ { \prime } ) =$ $0 \} | \ > \ \frac { n - 1 } { 2 }$ and $\overline { { u } } ( z ) ~ = ~ 1$ = 0 : = ˆ ( )otherwise. Clearly, the vector $\overline { y }$ 0 ( ) = 1given by this rule is exactly the same as the output vector of Algorithm 2.

We tried to use the power-iteration method in the Aggregation function for more than one step. However, the performance does not improve over the current version of Aggregation function based on majority vote. This is because in the spectral method above we tried our best to agree with $\hat { y } _ { / \mathbb { B } _ { 1 } } , \hat { y } _ { / \mathbb { B } _ { 2 } } , . . . , \hat { y } _ { / \mathbb { B } _ { n - 1 } } ,$ , ignoring the original channel ˆ ˆ ˆoutput y, and many of these are very noisy measurements.

## IV. DECODING ALGORITHM FOR GENERAL BINARY-INPUT MEMORYLESS CHANNELS

The decoding algorithm in the previous section only works for the BSC. In this section, we will present a natural extension of Algorithm 1 that works for any binary-input memoryless channels, and this new algorithm is based on LLRs (see (3)). Similarly to Algorithm 1, this new algorithm is also defined recursively, i.e., we first assume that we know how to decode r− -th order Reed-Muller code, and then we use it to decode ( 1)the r-th order Reed-Muller code. To begin with, note that the soft-decision FHT decoder [25] allows us to decode the first order RM code efficiently for general binary-input channels. The soft-decision FHT decoder is based on LLR, and the complexity is also O n n , the same as the hard-decision FHT decoder.

For completeness, we recap the FHT decoder in [25] for first order RM codes. We still use $c = ( c ( z ) , z \in \mathbb { E } )$ to denote =the transmitted (true) codeword and $y \ = \ ( y ( z ) , z \ \in \ \mathbb { E } )$ to = ( ( ) )denote the corresponding channel output. Given the output vector y, the ML decoder for first order RM codes aims to find $c \in \mathcal { R M } ( m , 1 )$ to maximize $\begin{array} { r } { \prod _ { z \in \mathbb { E } } W ( y ( z ) | c ( z ) ) } \end{array}$ . This is ( 1) ( ( )equivalent to maximizing the following quantity:

$$
\prod _ { z \in \mathbb { E } } \frac { W ( y ( z ) | c ( z ) ) } { \sqrt { W ( y ( z ) | 0 ) W ( y ( z ) | 1 ) } } ,
$$

which is further equivalent to maximizing

$$
\sum _ { z \in \mathbb { E } } \ln \Big ( \frac { W ( y ( z ) | c ( z ) ) } { \sqrt { W ( y ( z ) | 0 ) W ( y ( z ) | 1 ) } } \Big ) .\tag{8}
$$

Notice that the codeword c is a binary vector. Therefore,

$$
\begin{array} { r l } & { \quad \ln \Big ( \frac { W ( y ( z ) | c ( z ) ) } { \sqrt { W ( y ( z ) | 0 ) W ( y ( z ) | 1 ) } } \Big ) } \\ & { \quad = \left\{ \begin{array} { l l } { \frac { 1 } { 2 } \mathrm { L L R } ( y ( z ) ) } & { \mathrm { i f ~ } c ( z ) = 0 } \\ { - \frac { 1 } { 2 } \mathrm { L L R } ( y ( z ) ) } & { \mathrm { i f ~ } c ( z ) = 1 } \end{array} \right. } \end{array}
$$

From now on we will use the shorthand notation

$$
L ( z ) : = \operatorname { L L R } ( y ( z ) ) ,
$$

and the formula in (8) can be written as

$$
{ \frac { 1 } { 2 } } \sum _ { z \in \mathbb { E } } { \Big ( } ( - 1 ) ^ { c ( z ) } L ( z ) { \Big ) } ,\tag{9}
$$

so we want to find $c \in \mathcal { R M } ( m , 1 )$ to maximize this quantity. By definition, every $c \in \mathcal { R M } ( m , 1 )$ corresponds to a polynomial in $\mathbb { F } _ { 2 } [ Z _ { 1 } , Z _ { 2 } , \ldots , Z _ { m } ]$ ( 1)of degree one, so we can [ ]write every codeword c as a polynomial $\textstyle u _ { 0 } + \sum _ { i = 1 } ^ { m } u _ { i } Z _ { i }$ . In this way, we have $c ( z ) = u _ { 0 } + \textstyle \sum _ { i = 1 } ^ { m } u _ { i } z _ { i }$ +, where $z _ { 1 } , z _ { 2 } , \ldots , z _ { m }$ ( ) = +are the coordinates of the vector z. Now our task is to find $u _ { 0 } , u _ { 1 } , u _ { 2 } , \ldots , u _ { m } \in \mathbb { F } _ { 2 }$ to maximize

$$
\begin{array} { r l } & { ~ \displaystyle \sum _ { z \in \mathbb { E } } \Big ( ( - 1 ) ^ { u _ { 0 } + \sum _ { i = 1 } ^ { m } u _ { i } z _ { i } } L ( z ) \Big ) } \\ & { = ( - 1 ) ^ { u _ { 0 } } \displaystyle \sum _ { z \in \mathbb { E } } \Big ( ( - 1 ) ^ { \sum _ { i = 1 } ^ { m } u _ { i } z _ { i } } L ( z ) \Big ) . } \end{array}\tag{10}
$$

For a binary vector $\pmb { u } = ( u _ { 1 } , u _ { 2 } , \ldots , u _ { m } ) \in \mathbb { E }$ , we define

$$
\hat { L } ( \pmb { u } ) : = \sum _ { z \in \mathbb { E } } \Big ( ( - 1 ) ^ { \sum _ { i = 1 } ^ { m } u _ { i } z _ { i } } L ( z ) \Big ) .
$$

Clearly, to find the maximizer of (10), we only need to calculate $\hat { L } ( u )$ for all $u \in \mathbb { E }$ , but the vector $( \hat { L } ( { \pmb u } ) , { \pmb u } \in \mathbb { E } )$ is ( ) (exactly the Hadamard Transform of the vector $( L ( z ) , z \in \mathbb { E } )$ ， ( ( ) )so it can be calculated using the Fast Hadamard Transform with complexity $O ( n \log n )$ . Once we know the values of $( \hat { L } ( { \pmb u } ) , { \pmb u } \in \mathbb { E } )$ ( log ), we can find $\pmb { u } ^ { * } = ( u _ { 1 } ^ { * } , u _ { 2 } ^ { * } , \ldots , u _ { m } ^ { * } ) \in \mathbb { E }$ that ( ( )maximizes $| \hat { L } ( { \pmb u } ) |$ |. If $\hat { L } ( { \pmb u } ^ { * } ) > 0$ , then the decoder outputs ( ) ( )the codeword corresponding to $u _ { 0 } ^ { * } = 0 , u _ { 1 } ^ { * } , u _ { 2 } ^ { * } , \ldots , u _ { m } ^ { * }$ . Oth-= 0erwise, the decoder outputs the codeword corresponding to $u _ { 0 } ^ { * } = 1 , u _ { 1 } ^ { * } , u _ { 2 } ^ { * } , \ldots , u _ { m } ^ { * }$ . This completes the description of how = 1to decode the first order RM codes for general channels.

The next problem is how to extend (2) in the general setting. The purpose of (2) is mapping two output symbols $( y ( z ) , z \in$ T  whose indices are in the same coset $T \in \mathbb { E } / \dot { \mathbb { B } }$ to one )symbol. In this way, we reduce the r-th order RM code to an $( r - 1 )$ -th order RM code. For BSC, this mapping is simply ( 1)the addition in F2. The sum $y _ { / \mathbb { B } } ( T )$ can be interpreted as an estimate of $c _ { / \mathbb { B } } ( T )$ ( ), where c is the transmitted (true) codeword. (In other words,

$$
\begin{array} { r } { \mathbb { P } \big ( Y _ { / \mathbb { B } } ( T ) = c _ { / \mathbb { B } } ( T ) \big ) > \mathbb { P } \big ( Y _ { / \mathbb { B } } ( T ) = c _ { / \mathbb { B } } ( T ) \oplus 1 \big ) , } \end{array}
$$

where Y is the channel output random vector.

For general channels, we also want to estimate $c _ { / \mathbb { B } } ( T )$ based on the LLRs $( L ( z ) , z \ \in \ T )$ ( ). More precisely, given $( y ( z )$ , $z \in T )$ ( ( ) ), or equivalently given $( L ( z ) , z \in T )$ ( ( ), we would like to )calculate the following LLR:

$$
L _ { / \mathbb { B } } ( T ) : = \ln \Big ( \frac { \mathbb { P } \big ( Y ( z ) = y ( z ) , z \in T \big | c _ { / \mathbb { B } } ( T ) = 0 \big ) } { \mathbb { P } \big ( Y ( z ) = y ( z ) , z \in T \big | c _ { / \mathbb { B } } ( T ) = 1 \big ) } \Big ) .
$$

We will make use of the following simple property of RM codes to calculate this LLR.

Lemma 2: Suppose that $r \_ 1$ . Let C be a random codeword chosen uniformly from $\mathcal { R M } ( m , r )$ , and let z and $z ^ { \prime }$ ( )be two distinct vectors in E. Then the two coordinates $( C ( z ) , C ( z ^ { \prime } ) )$ of the random codeword C have i.i.d. Bernoulli-$1 / 2$ ) ( ))distribution.

Proof: Define the following four sets

$$
\begin{array} { r l } & { A ( 0 , 0 ) : = \{ c \in \mathcal { R M } ( m , r ) : c ( z ) = c ( z ^ { \prime } ) = 0 \} , } \\ & { A ( 0 , 1 ) : = \{ c \in \mathcal { R M } ( m , r ) : c ( z ) = 0 , c ( z ^ { \prime } ) = 1 \} , } \\ & { A ( 1 , 0 ) : = \{ c \in \mathcal { R M } ( m , r ) : c ( z ) = 1 , c ( z ^ { \prime } ) = 0 \} , } \\ & { A ( 1 , 1 ) : = \{ c \in \mathcal { R M } ( m , r ) : c ( z ) = c ( z ^ { \prime } ) = 1 \} . } \end{array}
$$

To prove this lemma, we only need to show that $| { \mathcal { A } } ( 0 , 0 ) | =$ $| \bar { \mathcal { A } ( 0 , 1 ) } | = | \bar { \mathcal { A } ( 1 , 0 ) } | = | \bar { \mathcal { A } ( 1 , 1 ) } |$ (0 0) =. Since RM code is linear and (0 1) = (1 0) = (1 1)the all one vector is a codeword of RM codes, the marginal distribution of the coordinate $C ( z )$ is Bernoulli- / for every $z \in \mathbb { E }$ . Thus we have

$$
\begin{array} { r } { \vert \boldsymbol { \mathcal { A } } ( 0 , 0 ) \vert + \vert \boldsymbol { \mathcal { A } } ( 0 , 1 ) \vert = \vert \boldsymbol { \mathcal { A } } ( 1 , 0 ) \vert + \vert \boldsymbol { \mathcal { A } } ( 1 , 1 ) \vert , } \\ { \vert \boldsymbol { \mathcal { A } } ( 0 , 0 ) \vert + \vert \boldsymbol { \mathcal { A } } ( 1 , 0 ) \vert = \vert \boldsymbol { \mathcal { A } } ( 0 , 1 ) \vert + \vert \boldsymbol { \mathcal { A } } ( 1 , 1 ) \vert . } \end{array}\tag{11}
$$

Now take $z = ( z _ { 1 } , \dots , z _ { m } )$ and $z ^ { \prime } = ( z _ { 1 } ^ { \prime } , \dots , z _ { m } ^ { \prime } )$ such that $z \neq z ^ { \prime } .$ = (. Then there exists $i \in [ m ]$ = (such that $z _ { i } \ \neq \ z _ { i } ^ { \prime } .$ . Since =we assume that $r \geq 1 , \mathcal { R M } ( m , r )$ =contains the evaluation 1 (vector of the degree- monomial $Z _ { i }$ ). We denote this evaluation 1vector as v, and we know that $\begin{array} { r } { v ( z ) \neq v ( z ^ { \prime } ) } \end{array}$ . Without loss of generality, assume that $v ( z ) = 0$ ( )and $v ( z ^ { \prime } ) = 1$ . Then we have4 $\mathcal { A } ( 0 , 0 ) + v \subseteq \mathcal { A } ( 0 , 1 )$ , so $| { \mathcal { A } } ( 0 , 0 ) | \leq | { \mathcal { A } } ( 0 , 1 )$ |. Conversely, (0 0) +we also have $\mathcal { A } ( 0 , 1 ) + v \subseteq \mathcal { A } ( 0 , 0 )$ , so $| \mathcal { A } ( 0 , 1 ) | \le | \mathcal { A } ( 0 , 0 )$ |. Therefore, $| \mathcal { A } ( 0 , 1 ) | = | \mathcal { A } ( 0 , 0 )$ 0 0) (0 1) (0 0)|. Similarly, we can also show that $| \mathcal { A } ( 1 , 1 ) | = | \mathcal { A } ( 1 , 0 )$ (0 0)|. Taking these into (11), we obtain that $| A ( 0 , 0 ) | = | A ( 0 , 1 ) | = | A ( 1 , 0 ) | = | A ( 1 , 1 ) |$ , which (0 0) = (0 1) = (completes the proof of the lemma. □

Now we can calculate $L _ { / \mathbb { B } } ( T )$ using the following model: Suppose that $S _ { 1 }$ and $S _ { 2 }$ ( )are i.i.d. Bernoulli- / random 1 2variables, and we transmit them over two independent copies of the channel $W : \{ 0 , 1 \} \to \mathcal { W } .$ . The corresponding channel : 0 1output random variables are denoted as $X _ { 1 }$ and $X _ { 2 }$ , respectively. Then for $x _ { 1 } , x _ { 2 } \ \in \ \mathcal { W }$ , we have (12), shown at the bottom of the page.

4For a set A and a vector v, we define the set $\mathcal { A } + v : = \{ a + v : a \in \mathcal { A } \}$

Lemma 2 above allows us to replace $x _ { 1 } , x _ { 2 }$ with $( y ( z )$ $z \in T )$ , and we obtain that

$$
L _ { / \mathbb { B } } ( T ) = \ln \Big ( \exp \big ( \sum _ { z \in T } L ( z ) \big ) + 1 \Big ) - \ln \Big ( \sum _ { z \in T } \exp ( L ( z ) ) \Big ) .\tag{13}
$$

Now we are ready to present the decoding algorithm for general binary-input channels. In Algorithms 3–4 below, we still denote the decoding result of the $( r - 1 )$ -th order RM code as $\hat { y } _ { / \mathbb { B } }$ ( 1)(see line  of Algorithm 3), where $\hat { y } _ { / \mathbb { B } } =$ $( \hat { y } _ { / \mathbb { B } } ( T ) , T \in \mathring { \mathbb { E } } / \mathbb { B } )$ are indexed by the cosets $T \in \mathbb { E } / \mathbb { B }$ , and we use $[ z + \mathbb { B } ]$ )to denote the coset containing z (see line  of [ + ]Algorithm 4).

Algorithm 3 is very similar to Algorithm 1: From line  to line 10, we compare $\hat { L } ( z )$ with the original $L ( z )$ 8. If the relative ( ) ( )difference between these two is below the threshold θ for every $z \in \mathbb { E }$ , then the values of $L ( z ) , z \in \mathbb { E }$ change very little in this ( )iteration, and the algorithm reaches a “stable" state, so we can exit the for loop on line 2. In practice, we find that $\theta = 0 . 0 5$ = 0 05works fairly well,5 and we still set the maximal number of iterations $N _ { \mathrm { m a x } } = m / 2$ , which is the same as in Algorithm 1. = 2On line 13, the algorithm simply produces the decoding result according to the LLR at each coordinate.

A few explanations of Algorithm 4: On line 3, we set c $\begin{array} { r } { \mathrm {  ~ \lambda ~ } \mathrm { \ m u L L R } ( z ) = \sum _ { z ^ { \prime } \ne z } \alpha ( z , z ^ { \prime } ) L ( z ^ { \prime } ) } \end{array}$ , where the coefficients $\alpha ( z , z ^ { \prime } )$ ( ) =can only be $\mathrm { ~ i ~ o r ~ } - 1$ ) ( ). More precisely, $\alpha ( z , z ^ { \prime } )$ is ( ) 1 1if the decoding result of the corresponding $( r \mathrm { ~ - ~ } 1 ) \mathrm { t h }$ ) 1 order RM code at the coset $\{ z , z ^ { \prime } \}$ is , and $\alpha ( z , z ^ { \prime } )$ 1)is − if the decoding result at the coset $\{ z , z ^ { \prime } \}$ ( ) 1is . The reason behind 1this assignment is simple: The decoding result at the coset $\{ z , z ^ { \prime } \}$ is an estimate of $c ( z ) \oplus c ( z ^ { \prime } )$ . If $c ( z ) \oplus c ( z ^ { \prime } )$ is more ( )likely to be , then the sign of $L ( z )$ and $L ( z ^ { \prime } )$ should be the 0 ( ) ( )same. Here cumuLLR z serves as an estimate of $L ( z )$ based on all the other $L ( z ^ { \prime } ) , z ^ { \prime } \neq z ,$ ( )so we assign the coefficient $\alpha ( z , z ^ { \prime } )$ to be . Otherwise, if $c ( z ) \oplus c ( z ^ { \prime } )$ is more likely to be ( ) 1, then the sign of $L ( z )$ and $L ( z ^ { \prime } )$ ( )should be different, so we 1assign the coefficient $\alpha ( z , z ^ { \prime } )$ ( )to be − .

( ) 1In Algorithms 3–4, we write the pseudo codes in a mathematical fashion for the ease of understanding. In Appendix $\mathrm { C } ,$ we present another version of the RPA\_RM function in a program language fashion; see Algorithm 9.

$$
\begin{array} { r l } & { \quad \ln \Big ( \frac { \mathbb { P } ( X _ { 1 } = x _ { 1 } , X _ { 2 } = x _ { 2 } | S _ { 1 } + S _ { 2 } = 0 ) } { \mathbb { P } ( X _ { 1 } = x _ { 1 } , X _ { 2 } = x _ { 2 } | S _ { 1 } + S _ { 2 } = 1 ) } \Big ) = \ln \Big ( \frac { \mathbb { P } ( X _ { 1 } = x _ { 1 } , X _ { 2 } = x _ { 2 } , S _ { 1 } + S _ { 2 } = 0 ) } { \mathbb { P } ( X _ { 1 } = x _ { 1 } , X _ { 2 } = x _ { 2 } , S _ { 1 } + S _ { 2 } = 1 ) } \Big ) } \\ & { = \ln \Big ( \frac { \mathbb { P } ( X _ { 1 } = x _ { 1 } , X _ { 2 } = x _ { 2 } , S _ { 1 } = 0 , S _ { 2 } = 0 ) + \mathbb { P } ( X _ { 1 } = x _ { 1 } , X _ { 2 } = x _ { 2 } , S _ { 1 } = 1 , S _ { 2 } = 1 ) } { \mathbb { P } ( X _ { 1 } = x _ { 1 } , X _ { 2 } = x _ { 2 } , S _ { 1 } = 1 , S _ { 2 } = 0 ) } \Big ) } \\ & { = \ln \Big ( \frac { \frac { 1 } { 4 } W ( x _ { 1 } | 0 ) W ( x _ { 2 } | 0 ) + \frac { 1 } { 4 } W ( x _ { 1 } | 1 ) W ( x _ { 2 } | 1 ) } { \mathbb { P } ( x _ { 1 } | 0 ) } \Big ) = \ln \Big ( \frac { W ( x _ { 1 } | 0 ) W ( x _ { 2 } | 0 ) } { \frac { W ( x _ { 1 } | 1 ) W ( x _ { 2 } | 1 ) } { W ( x _ { 1 } | 1 ) } + \frac { 1 } { W ( x _ { 2 } | 1 ) } } \Big ) } \\ &  = \ln \Big ( \frac { 1 } { 4 } W ( x _ { 1 } | 0 ) W ( x _ { 2 } | 1 ) + \frac { 1 } { 4 } W ( x _ { 1 } | 1 ) W ( x _ { 2 } | 0 ) \Big ) = \ln \Big ( \frac  W ( x _  1 \end{array}\tag{12}
$$

Algorithm 3 The RPA\_RM Decoding Function for General   
Binary-Input Memoryless Channels   
Input: The LLR vector $( L ( z ) , z \in \{ 0 , 1 \} ^ { m } )$ ; the parameters   
( ( )of the Reed-Muller code m and $r ;$ 0 1 ) the maximal number of   
iterations $N _ { \mathrm { m a x } } ;$ the exiting threshold θ   
Output: The decoded codeword $\hat { c }$   
$\mathbf { \mathbb { 1 } } \colon \mathbb { E } : = \{ 0 , 1 \} ^ { m }$   
:2: for $j = 1 , 2 , \dots , N _ { \mathrm { m a x } }$ do   
3: $L _ { / \mathbb { B } _ { i } } \gets ( L _ { / \mathbb { B } _ { i } } ( T ) , T \in \mathbb { E } / \mathbb { B } _ { i } ) \mathrm { ~ f o r ~ } i = 1 , 2 , \dots , 2 ^ { m } - 1$   
4: - $L _ { / \mathbb { B } _ { i } } ( T )$ ( ) )is calculated from $( L ( z ) , z \in \mathbb { E } )$ 2 1according   
to (13)   
5: $\boldsymbol { \hat { y } } _ { / \mathbb { B } _ { i } } \gets \mathrm { R P } \mathbb { A } _ { - } \mathrm { R M } ( L _ { / \mathbb { B } _ { i } } , m - 1 , r - 1 , N _ { \operatorname* { m a x } } , \theta )$ for $i =$   
, $, 2 , \ldots , 2 ^ { m } - 1$   
6: $\triangleright \operatorname { I f } r = 2 ,$ 1, then we use the Fast Hadamard Transform   
= 2to decode the first-order RM code   
7: $\hat { L } \gets \mathsf { A g g r e g a t i o n } ( L , \hat { y } _ { / \mathbb { B } _ { 1 } } , \hat { y } _ { / \mathbb { B } _ { 2 } } \ldots , \hat { y } _ { / \mathbb { B } _ { n - 1 } } )$   
8: if $| \hat { L } ( z ) - L ( z ) | \leq \theta | L ( z ) |$ ˆfor all $z \in \mathbb { E }$ then - The   
( ) ( ) ( )algorithm reaches a stable point   
9: break   
10: end if   
11: $L \gets \hat { L }$   
12: end for   
13: $\hat { c } ( z ) \gets \mathbb { 1 } [ L ( z ) < 0 ]$ for each $z \in \mathbb { E }$   
ˆ( )14: return c

```latex
Algorithm 4 The Aggregation Function for General
Binary-Input Memoryless Channels
Input: $\overline { { L , \hat { y } _ { / \mathbb { B } _ { 1 } } , \hat { y } _ { / \mathbb { B } _ { 2 } } \ldots , \hat { y } _ { / \mathbb { B } _ { n - 1 } } } }$
ˆOutput: L
1: Initialize $( \mathsf { c u m u L L R } ( z ) , z \in \{ 0 , 1 \} ^ { m } )$ as an all-zero vector
(indexed by $z \in \{ 0 , 1 \} ^ { m }$
2: $n \gets 2 ^ { m }$
23: cumuLL $\begin{array} { r } { \mathfrak { a } ( z ) \longleftarrow \sum _ { i = 1 } ^ { n - 1 } \big ( ( 1 - 2 \hat { y } _ { / \mathbb { B } _ { i } } ( [ z + \mathbb { B } _ { i } ] ) ) L ( z \oplus z _ { i } ) \big ) } \end{array}$ for
each $z \in \{ 0 , 1 \} ^ { m }$
4: $\triangleright z _ { i }$ is the nonzero element in $\mathbb { B } _ { i }$
5: $\triangleright \hat { y } _ { / \mathbb { B } _ { i } }$ is the decoded codeword, so $\hat { y } _ { / \mathbb { B } _ { i } } ( [ z + \mathbb { B } _ { i } ] )$ is
ˆeither  or
6: $\begin{array} { r } { \hat { L } ( z ) \longleftarrow \frac { \mathtt { c u m u L L R } ( z ) } { n - 1 } } \end{array}$ for each $z \in \{ 0 , 1 \} ^ { m }$
7: return $\hat { L }$
```

Following the same proof of Proposition 1, we have the following result

:Proposition 3: The complexity of Algorithm 3 is $O ( n ^ { r } \log { n } )$ in sequential implementation and $O ( n ^ { 2 } )$ in ( log )parallel implementation with $O ( n ^ { r } )$ processors.

( )In Section V, we present an accelerated version of the RPA algorithm for high-rate RM codes, and in Section VI-C, we further discuss other possible options to reduce the computation time by using fewer subspaces in the projection step.

Similarly to Proposition 2, we can also show that the decoding error probability of Algorithm 3 is independent of the transmitted codeword for binary-input memoryless symmetric (BMS) channels.

Definition 3 (BMS channel): We say that a memoryless channel $W : \{ 0 , 1 \} \to \mathcal { W }$ is a BMS channel if there is a permutation π of the output alphabet W such that $\pi ^ { - 1 } = \pi$ and $W ( x | 1 ) = W ( \pi ( x ) | 0 )$ for all $x \in \mathcal { W }$

( 1) = ( (Proposition 4: Let $W : \{ 0 , 1 \} \to \mathcal { W }$ be a BMS channel. Let $c _ { 1 }$ and $c _ { 2 }$ : 0 1be two codewords of $\mathcal { R M } ( m , r )$ . Let $Y _ { 1 }$ and $Y _ { 2 }$ ( )be the (random) channel outputs of transmitting $c _ { 1 }$ and c2 over $n = 2 ^ { m }$ independent copies of W , respectively. Let $L ^ { ( 1 ) }$ and $L ^ { ( 2 ) }$ 2be the LLR vectors corresponding to $Y _ { 1 }$ and $Y _ { 2 } ,$ respectively.6 Then for any $c _ { 1 } , c _ { 2 } \in \mathcal { R M } ( m , r )$ , we have

The proof is given in Appendix B. Similarly to Proposition 2, this proposition is also very useful for simulations because we can simply transmit the all-zero codeword over the BMS channel W to measure the decoding error probability.

In the last part of this section, we present the list decoding version of the RPA\_RM function. The main idea is already explained in Section II-A. Here we only write down the pseudo code of the list decoding version. Note that the purpose of line  is to make sure that $\hat { c } ^ { ( u ) }$ is a codeword of RM code, which is not always true for the decoding result of the RPA\_RM function.

Finally, we present the following proposition on the memory requirement for sequential implementation of RPA decoder. A remarkable thing here is that the memory requirement for the list decoding version of RPA algorithm is $5 n ,$ which is 5independent of the list size, in contrast to SCL decoder of polar codes.

Proposition 5: The memory needed for sequential implementation of the RPA decoder without list decoding is no more than n, and the memory needed for sequential implementa-4tion of the RPA decoder with list decoding is no more than n, 5where n is the code length. Note that the memory requirement for list decoding version does not depend on the list size.

Proof: As we mentioned above, Algorithm 3 is written in compact fashion for the ease of understanding, but it is not space-efficient in practical implementation. The version that we really implemented in practice and used for simulations is Algorithm 9 in Appendix $\mathrm { C } ,$ and our analysis of space complexity is based on Algorithm 9.

The most important difference between Algorithm 9 and Algorithm 3 is that in Algorithm 3 we first finish all the recursive decoding and then perform the aggregation step; while in Algorithm 9 the recursive decoding step and the aggregation step are interleaved together, and in this way we can save huge amount of memory compared to Algorithm 3.

We start with RPA decoder without list decoding, and we prove by induction on r, the order of the RM code. For the base case of $r \ = \ 1$ , the claim clearly holds. Now assume = 1that the claim holds for all RM codes with order $< r$ and we prove it for order r. In Algorithm 9, we need n floating number positions to store the LLR vector and another n floating number positions to store the cumuLLR vector. Then we project onto the cosets of each one-dimensional subspace sequentially. For each projected codeword, we need to decode a RM code with length $n / 2$ and order $r - 1$ . By induction hypothesis, this take ∗ $n / 2 = 2 n$ 1floating number positions. 4Therefore in total we need $n + n + 2 n = 4 n$ floating number + + 2 = 4positions. This establishes the inductive step and completes the proof for the non-list-decoding version.

Algorithm 5 The RPA\_LIST Decoding Function for General   
Binary-Input Memoryless Channels   
Input: The LLR vector $( L ( z ) , z \in \{ 0 , 1 \} ^ { m } )$ ; the parameters   
( ( ) 0 1 )of the Reed-Muller code m and r; the maximal number of   
iterations $N _ { \mathrm { m a x } } ;$ the exiting threshold $\theta ;$ the list size $2 ^ { t }$   
Output: The decoded codeword c   
1: $\tilde { L } \gets L$   
2: $( z _ { 1 } , z _ { 2 } , \ldots , z _ { t } ) \quad $ indices of the t smallest entries in   
$( | L ( z ) | , z \in \{ 0 , 1 \} ^ { m } )$   
3: $\textsf { \textsf { P } } z _ { i } \in \{ 0 , 1 \} ^ { m }$ for all $i = 1 , 2 , \dots , t$   
4: $L _ { \mathrm { m a x } } \gets 2$ max $( | L ( z ) | , z \in \{ 0 , 1 \} ^ { m } )$   
5: for each $\pmb { u } \in \{ L _ { \mathrm { m a x } } , - L _ { \mathrm { m a x } } \} ^ { t }$ 1do   
6: $( L ( z _ { 1 } ) , L ( z _ { 2 } ) , \ldots , L ( z _ { t } ) )  u$   
7: $\hat { c } ^ { ( \pmb { u } ) } \gets \mathrm { R P A \_ R M } ( L , m , r , N _ { \mathrm { m a x } } , \theta )$   
8: $\hat { c } ^ { ( u ) } \gets \mathrm { R e e d s d e c o d e r } ( \hat { c } ^ { ( u ) } )$ - Reedsdecoder is   
ˆ (ˆ )the classical decoding algorithm in [2]   
9: end for   
10: $\begin{array} { r } { u ^ { * } \gets \operatorname * { a r g m a x } _ { u } \sum _ { z \in \{ 0 , 1 \} ^ { m } } \left( ( - 1 ) ^ { \hat { c } ^ { ( u ) } ( z ) } \tilde { L } ( z ) \right) } \end{array}$   
11: - This follows from (9). Maximization is over   
$\pmb { u } \in \{ L _ { \operatorname* { m a x } } , - L _ { \operatorname* { m a x } } \} ^ { t }$   
12: $\hat { c } \gets \hat { c } ^ { ( { \boldsymbol u } ^ { * } ) }$   
ˆ ˆ13: return c

The memory requirement for list decoding version follows directly from that of the vanilla version: Since we perform list decoding sequentially, i.e., we only decode one list at a time, the only extra memory we need in the list decoding version is the n floating number positions that is used to store currently best known decoding result. Therefore, the space complexity for the list decoding version is n. □

## V. SIMPLIFIED RPA ALGORITHM FOR HIGH RATE RM CODES

In this section, we provide some simplified versions of the RPA decoder, which significantly accelerate the decoding process while maintaining the same (nearly optimal) decoding error probability for certain RM codes with rate $> 0 . 5$

As mentioned in the previous section, we can accelerate the decoding algorithm by using fewer subspaces in the projection step. Moreover, instead of using one-dimensional subspaces, in this section we propose to use a selected subsets of two-dimensional subspaces in the projection step. In particular, we only project onto the $\binom m 2$ two-dimensional subspaces spanned by two standard basis vectors of E. The standard basis vector of E are $\pmb { e } ^ { ( 1 ) } , \ldots , \pmb { e } ^ { ( m ) }$ , where $e ^ { ( i ) }$ is defined as the vector with  in the ith position and  everywhere 1else. Then we write the $\binom m 2$ 0two-dimensional subspaces as $\{ \mathbb { B } _ { i , j } : 1 \leq i < j \leq m \}$ , where

$$
\mathbb { B } _ { i , j } : = \operatorname { s p a n } ( e ^ { ( i ) } , e ^ { ( j ) } ) .
$$

Note that projection onto cosets of two-dimensional subspaces is different from onto that of one-dimensional subspaces: In the one-dimensional case, each coset only contains two coordinates, and we only need to combine the LLR of two coordinates to obtain the LLR of the coset, as we did in (13). In the two-dimensional case, each coset contains four coordinates, and we need to combine the LLR of four coordinates to obtain the LLR of the coset. Fortunately, for any RM code with order $r \geq 2 ,$ , we can use exactly the same idea in the proof of Lemma 2 to show that any four coordinates in a coset of a two-dimensional subspace are also independent; see the explanation in Remark 1 below. Therefore, we obtain the following counterpart of (13) for a coset T of two-dimensional subspace assuming that $T = \{ z ^ { ( 1 ) } , z ^ { ( 2 ) } , z ^ { ( 3 ) } , z ^ { ( 4 ) } \}$ :

$$
\begin{array} { l } { { \displaystyle { \cal L } _ { / \mathbb { B } } ( T ) = \ln \Big ( \exp \big ( \displaystyle \sum _ { i = 1 } ^ { 4 } L ( z ^ { ( i ) } ) \big ) } \ ~ } \\ { { \displaystyle ~ + \sum _ { 1 \leq i < j \leq 4 } \exp \big ( L ( z ^ { ( i ) } ) + L ( z ^ { ( j ) } ) \big ) + 1 \Big ) } } \\ { { \displaystyle ~ - \ln \Big ( \displaystyle \sum _ { i = 1 } ^ { 4 } \exp ( L ( z ^ { ( i ) } ) ) + \displaystyle \sum _ { i = 1 } ^ { 4 } \exp ( \displaystyle \sum _ { j \in [ 4 ] \backslash \{ i \} } L ( z ^ { ( j ) } ) ) \Big ) . } } \end{array}\tag{14}
$$

Remark 1: It is well known that for a linear code, if there is a codeword taking value  at a certain coordinate, then 1the number of codewords taking value at this coordinate 1is the same as the number of codewords taking value  at 0this coordinate. This follows directly from the linearity of the code. The proof of Lemma 2 follows from the same idea: By the linearity of code, we only need to show that for two distinct coordinates, there are different codewords in RM codes that take all four possible values $( 0 , 0 ) , ( 0 , 1 ) , ( 1 , 0 ) , ( 1 , 1 )$ at (0 0) (0 1) (1 0) (1 1)these two coordinates, and this follows by noting that (i) any two distinct coordinates form a coset of a one-dimensional subspace; (ii) by definition of RM codes, restricting RM codes with order $r \geq 1$ on such cosets gives us $\mathcal { R M } ( 1 , 1 )$ 1 (1 1)which contains all  binary vectors of length . Now in 4 2the case of two-dimensional subspace, we still use the same reasoning: By linearity of the code, we only need to show that for any coordinates that form a coset of a 2-dimensional 4subspace, there are different codewords in RM codes with order $r \geq 2$ that take all $2 ^ { 4 }$ possible values $\{ 0 , 1 \} ^ { 4 }$ at these 2 2 0 1four coordinates. This again follows by noting that restricting RM codes with order $r \geq 2$ on such cosets gives us $\mathcal { R M } ( 2 , 2 )$ . 2which contains all binary vectors of length .

After projecting $\mathcal { R M } ( m , r )$ 4onto the cosets of these ( )two-dimensional subspaces, we will obtain RM codes with parameters $m - 2$ and $r \mathrm { ~ - ~ } 2 ,$ , as proved in Lemma 1. After 2 2decoding these m2  projected codes $\mathcal { R M } ( m - 2 , r - 2 )$ ， we obtain $\{ \hat { y } _ { / \mathbb { B } _ { i , j } } : \ 1 \ \le \ i < \ j \ \le \ m \}$ (, where $\begin{array} { r l } { \hat { y } _ { / \mathbb { B } _ { i , j } } } & { { } = } \end{array}$ $( \hat { y } _ { / \mathbb { B } _ { i , j } } ( T ) , T \in \mathbb { E } / \mathbb { B } _ { i , j } )$ ˆ =. Now we are ready to go to the (ˆ ( ) )aggregation step using both the recursive decoding result $\{ \hat { y } _ { / \mathbb { B } _ { i , j } } : 1 \le i < j \le m \}$ and the original LLR vector $L .$ ˆ : 1In particular, when decoding c z , the relevant coordinate in $\hat { y } _ { / \mathbb { B } _ { i , j } }$ is $\hat { y } _ { / \mathbb { B } _ { i , j } } ( [ z + \mathbb { B } _ { i , j } ] )$ ( ), where $\left[ z + \mathbb { B } _ { i , j } \right]$ is the coset of $\mathbb { B } _ { i , j }$ ˆ ([ + ]) [ + ]that contains z. Now suppose that the other three vectors in $\left[ \boldsymbol { z } + \mathbb { B } _ { i , j } \right]$ apart from z itself are $z ^ { ( 1 ) } , z ^ { ( 2 ) } , z ^ { ( 3 ) }$ . Then from $\hat { y } _ { / \mathbb { B } _ { i , j } } ( [ z + \mathbb { B } _ { i , j } ] )$ and $L ( z ^ { ( 1 ) } ) , L ( z ^ { ( 2 ) } ) , L ( z ^ { ( 3 ) } )$ , we obtain the following estimate of the LLR of $c ( z ) { \mathrm { ; } }$

$$
\begin{array} { r } { \mathrm { E s t } _ { i , j } ( z ) = \ln \Big ( \exp \big ( \displaystyle \sum _ { i = 1 } ^ { 3 } L ( z ^ { ( i ) } ) \big ) + \displaystyle \sum _ { i = 1 } ^ { 3 } \exp ( L ( z ^ { ( i ) } ) ) \Big ) } \\ { \displaystyle - \ln \Big ( \displaystyle \sum _ { i = 1 } ^ { 3 } \exp \big ( \displaystyle \sum _ { j \geq 3 \mid i \leq i } L ( z ^ { ( i ) } ) \big ) + 1 \Big ) } \\ { \displaystyle + \ln \big ( \displaystyle \sum _ { i = 1 } ^ { 3 } L ( z ^ { ( i ) } ) \big ) - 0 , } \\ { \mathrm { E s t } _ { i , j } ( z ) = - \ln \Big ( \exp \big ( \displaystyle \sum _ { i = 1 } ^ { 3 } L ( z ^ { ( i ) } ) \big ) + \displaystyle \sum _ { i = 1 } ^ { 3 } \exp ( L ( z ^ { ( i ) } ) ) \Big ) } \\ { \displaystyle + \ln \Big ( \displaystyle \sum _ { i = 1 } ^ { 3 } \exp ( \displaystyle \sum _ { j \geq 3 \mid i \leq i } L ( z ^ { ( i ) } ) ) + 1 \Big ) } \\ { \displaystyle + \ln \Big ( \displaystyle \sum _ { i = 1 } ^ { 3 } \exp ( \displaystyle \sum _ { j \geq 3 \mid i \leq i } L ( z ^ { ( i ) } ) ) + 1 \Big ) } \\ { \displaystyle + \ln ( \displaystyle \sum _ { i = 1 } ^ { 4 } \exp ( \exp ( \exp ( \exp ( \xi ) ) ) - 1 , } \\ { \displaystyle + \ln ( \displaystyle \sum _ { i = 1 } ^ { 3 } L ( z ^ { ( i ) } ) \big ) - 1 , ) } \end{array}\tag{15}
$$

We calculate such an estimate for all pairs of $( i , j )$ such that $1 \leq i < j \leq m$ . Then finally we update the LLR of $c ( z )$ as 1the average of these $\binom m 2$ estimates, as follows:

$$
\hat { L } ( z ) = \frac { 1 } { { \binom { m } { 2 } } } \sum _ { 1 \leq i < j \leq m } \mathrm { E s t } _ { i , j } ( z ) .
$$

Finally, as in all the previous sections, we iterate this decoding procedure a few times for the LLR vector to converge to a stable value.

We call the decoding algorithm proposed in this section the Simplified\_RPA algorithm, as opposed to the normal RPA algorithm proposed in the previous section. Note here that in the recursive decoding procedure, i.e., when we decode $\mathscr { R M } ( m - 2 , r - 2 )$ , we still use this simplified version of RPA algorithm instead of doing full projection step. Since each time we reduce r by , if the original r is even then we will not reach the first-order RM codes. In this case, we use the normal RPA decoder when we reach the second-order RM codes. In Algorithm 6 and Algorithm 7 we provide pseudo-codes for the Simplified\_RPA algorithm. Note that in line 7–8 of Algorithm 6, we distinguish between the cases of r being even and r being odd: For even r, eventually we will need to decode a second-order RM code using the normal RPA decoder while for odd r, we only need to decode first-order RM code in the final recursive step. As we will show in Section VI (see Fig. 2), by applying the list decoding version of the Simplified\_RPA algorithm, we can decode $\mathcal { R M } ( 7 , 4 )$ and $\mathcal { R M } ( 8 , 5 )$ (7 4)with list size no larger than  such that the decoding (8 5) 8error probability is the same as that of ML decoder. Moreover, it runs even faster than decoding lower rate codes such as ${ \mathcal { R M } } ( 8 , 3 )$ ; see Table I.

## VI. SIMULATION RESULTS

## A. Comparison With Polar Codes

We run our decoding algorithm for second and third order Reed-Muller codes with code length ,  and  over 256 512 1024AWGN channels and BSCs, and we compare its performance with the recent algorithms for polar codes with the same length and dimension. We compare to two versions of polar codes: Polar codes with optimal CRC size and polar codes without

Algorithm 6 The Simplified\_RPA Decoding Function   
Input: The LLR vector $\overline { { ( L ( z ) , z \in \{ 0 , 1 \} ^ { m } ) } }$ ; the parameters   
( ( ) 0 1 )of the Reed-Muller code m and r; the maximal number of   
iterations $N _ { \mathrm { m a x } } ;$ ; the exiting threshold θ   
Output: The decoded codeword c   
1: E  { , }m   
:2: for $j = 1 , 2 , \dots , N _ { \mathrm { m a x } }$ do   
3: $L _ { / \mathbb { B } _ { i , j } } \gets ( L _ { / \mathbb { B } _ { i , j } } ( T ) , T \in \mathbb { E } / \mathbb { B } _ { i , j } )$ for $1 \leq i < j \leq m$   
4: - $\cdot \ \bar { L _ { / \mathbb { B } _ { i , j } } } \left( T \right)$ ) 1is calculated according to (14)   
5: $\begin{array} { r l r } { \hat { y } _ { / \mathbb { B } _ { i , j } } } & { { }  } & { \mathsf { S i m p l i f i e d \_ R P A } ( L _ { / \mathbb { B } _ { i , j } } , m \mathrm { ~ - ~ } 2 , r \mathrm { ~ - ~ } } \end{array}$   
, $N _ { \mathrm { m a x } } , \theta )$ for $1 \leq i < j \leq m$   
6: $\triangleright \operatorname { I f } r = 3 ,$ 1 then we use the Fast Hadamard Transform   
= 3to decode the first-order RM code   
7: - If r  , then we use the normal RPA algorithm to   
= 4decode the second-order RM code   
8: L ← Simp\_Aggregation $( L , \{ \hat { y } _ { / \mathbb { B } _ { i , j } } : 1 \leq i < j \leq$   
m}   
)9: if $| \hat { L } ( z ) - L ( z ) | \leq \theta | L ( z ) |$ for all $z \in \mathbb { E }$ then - The   
( ) ( ) ( )algorithm reaches a stable point   
10: break   
11: end if   
12: $L \gets \hat { L }$   
13: end for   
14: $\hat { c } ( z ) \gets \mathbb { 1 } [ L ( z ) < 0 ]$ for each $z \in \mathbb { E }$   
ˆ( ) [15: return c   
Algorithm 7 The Simp\_Aggregation Function in the   
Simplified\_RPA Algorithm   
Input: $L , \{ \hat { y } _ { / \mathbb { B } _ { i , j } } : 1 \leq i < j \leq m \}$   
Output: L   
1: Calculate $\operatorname { E s t } _ { i , j } ( z )$ from L and $\{ \hat { y } _ { / \mathbb { B } _ { i , j } } : 1 \le i < j \le m \}$   
Est (according to (15)   
2: $\begin{array} { r } { \hat { L } ( z ) \gets \frac { 1 } { \binom { m } { 2 } } \sum _ { 1 \leq i < j \leq m } \mathrm { E s t } _ { i , j } ( z ) } \end{array}$ for each $z \in \{ 0 , 1 \} ^ { m }$   
(3: return L   
TABLE I   
COMPARISON OF DECODING TIME BETWEEN RM CODES AND POLAR   
CODES. $P ( m , r )$ DENOTES POLAR CODES WITH THE SAME   
LENGTH AND DIMENSION AS $\mathcal { R M } ( m , r )$   
RM(7,2) $\overline { { P ( 7 , 2 ) } }$ RM(7,3) $\underline { { \overline { { P ( 7 , 3 ) } } } }$ RM(7,4) P(7,4)   
1ms 7ms 26ms 15ms 6ms 23ms   
RM(8,2) P(8,2) RM(8,3) ${ \overline { { P ( 8 , 3 ) } } }$ RM(8,4) P(8,4   
4.3ms 17ms 236ms 140ms 5.9s 64ms   
RM(8,5) P(8,5) RM(9,2) P(9,2) RM(10,2) P(10,2)   
14ms 82ms 18.2ms 41ms 76.7ms 95ms

CRC, and we use the Successive Cancellation List (SCL) decoder introduced by Tal and Vardy [20] as the decoder, where we set list size to be . Note that SCL decoder with 32list size  is one of the most widely used decoders for polar codes.

The simulation results for AWGN channels are plotted in Figure 2, where the number of Monte Carlo trials is 5. We provide the simulation results for 10all RM codes with length 128 and 256, including RM , , RM , , RM , , RM , , RM , , (7 2) (7 3) (7 4) (8 2) (8 3) [4]RM , , RM , . This should give a complete picture (8 4) (8 5)of the performance of our decoder for all code rates. Note that we skipped RM ,  and RM ,  because they are (7 5) (8 6)extended Hamming codes, and optimal decoders are well known for these two codes. Moreover, for certain cases the list decoding version of RPA decoding algorithm has almost the same performance as the Maximal Likelihood (ML) decoder for RM codes.7 The performance improvement is thus in agreement with the advantages of RM codes over polar codes under ML decoding [5]. See Section VI-B for comparisons with Dumer’s recursive decoding algorithm [14]–[16], which is the best known decoder in the literature for RM codes over AWGN channels. Note also that the algorithm in [19] only applies to codes with very short code length (no larger than ) due to complexity constraints.

![](images/97428d627e0c0bb14595823ceb866e89d6b4ca603ee1d0c951a94afb8176b4ca.jpg)  
(a) RM(7,2) v.s. polar codes

![](images/9eeec1df39ce0cc4d455285981879d5141c510f957a3f68faa16644584a20230.jpg)  
(b) $\mathcal { R M } ( 7 , 3 )$ v.s. polar codes

![](images/71d012b703e76fb22561abea8b4e8274ca680f9dcdb56a4e2a9f473a9cdae630.jpg)  
（c） $\mathcal { R M } ( 7 , 4 )$ v.s. polar codes

![](images/a1fc7467468dbf7cd2bfb4609bd9d7248b856b4b5af1b1bfd55564c7c9fd139b.jpg)  
(d) RM(8,2) v.s. polar codes

![](images/96c40dcfd90e0344fb4d0538bcbe9206eeffb76a139a8c2844e5cdbd310dc660.jpg)

![](images/0699acf1cd3eea30db1d9b3953b4e99341844a731f60857b0a8978f938e1daa3.jpg)

(f) ${ \mathcal { R M } } ( 8 , 4 )$ v.s. polar codes  
(e) RM(8,3) v.s. polar codes  
![](images/fec8f57c49ea9c26e7fa10a990b0295eaca51c56640f4f369b726dc3470abf2f.jpg)  
(g) RM(8,5) v.s. polar codes

![](images/72e08d2d1f63b03c476eb7a7bfb129a45d2623ee599e5521a3b0236482e23811.jpg)

![](images/7453513dcbde10907585e3441181a050a9a196530945d2b889350780ca2c48f3.jpg)  
(h) RM(9,2) v.s. polar codes  
(i) RM(10,2) v.s. polar codes  
Fig. 2. Comparison between Reed-Muller codes and polar codes over AWGN channels. For RM(7, 4) and RM(8, 5), we use the Simplified\_RPA algorithm proposed in Section V, and for all the other RM codes, we use the normal RPA algorithm proposed in Section IV. For polar codes with or without CRC, we always use SCL decoder with list size 32. For polar codes with CRC, we test various choices of CRC length and choose the optimal one that gives the best performance. The number in the bracket after “Polar-CRC” is the optimal CRC length that we use.

![](images/b83b7d16ebb0c989543c2ccf4eb9eaeee6d08da9450bfae5f6204917e54be8e3.jpg)  
(a) RM(8,2) vs Polar codes

![](images/38d4530733e2f07aa4cef9990e7b07ed6a95bccd19ac57a9438086eba1755816.jpg)  
(b) RM(8,3) vs Polar codes

![](images/63de0a1818dbcfca7d7c76f4859d95de80b0ee4d0f53648f8c27718d61583d10.jpg)  
(c) RM(9,2) vs Polar codes  
Fig. 3. Comparison between Reed-Muller codes and polar codes over BSC channels. For RM codes we use the RPA decoder in Algorithm 1 without list decoding. For polar codes, no matter with or without CRC, we always use SCL decoder with list size 32.

28For the BSC channel, the simulation results are plotted in Figure 3. The number of Monte Carlo trials is 5. 10We also tested in this case all the previous decoding algorithms known for RM codes, including Reed’s algorithm [2] and the algorithm from Saptharishi-Shpilka-Volk [17]. For these two algorithms, the decoding error probability exceeds . for the tested parameters, so we did not include them 0 1in Figure 3 as they would not fit. See Section VI-B for comparisons with the Sidel’nikov-Pershakov algorithm [11] and its variations [12], [13]. From Figure 3, we can clearly see that the new decoding algorithm for RM codes significantly outperforms the SCL decoder for CRC-aided polar codes.

We also compare the running time of our decoder and the SCL decoder for polar codes. For polar codes, we use techniques from two accelerated version [26], [27] of the SCL decoder (in particular the “min-sum approximation" in [26]) so that we can achieve a much smaller running time than the original version of SCL decoder while maintaining almost the same decoding error probability. The results are listed in Table I. We can see that for second order RM codes as well as the high-rate RM codes where we use the Simplified\_RPA algorithm to decode, our decoder is always faster than the SCL decoder for polar codes with the same parameters. However, for third order RM codes, our decoder is slower than the SCL decoder; see Fig 2 for decoding error probability and Table I for running time.

## B. Comparison With Previous Decoding Algorithms of RM Codes

We first compare with the decoding algorithm proposed by Sidel’nikov and Pershakov [11], which was later improved/modified in [12], [13]. When decoding the second-order RM codes, the RPA decoding algorithm has some high-level similarity with the decoding algorithms in [11]–[13] in the sense that the first step in all these algorithms is to project the received word y onto the cosets of all the n− one-dimensional subspaces and decode the projected 1first-order RM codewords to obtain $\hat { y } _ { / \mathbb { B } _ { 1 } } , \hat { y } _ { / \mathbb { B } _ { 2 } } , \dots , \hat { y } _ { / \mathbb { B } _ { n - 1 } } .$ ˆ ˆ ˆHowever, the next steps in [11]–[13] are quite different from the RPA decoding algorithm and result in a worse performance than the RPA algorithm. More precisely, the main differences are

:• The decoding algorithms in [12], [13] only work for the second order RM codes. For higher-order RM codes, the decoding algorithm proposed in [11] is completely different from the RPA algorithm, and their performance is much worse than the RPA algorithm; see Fig. 4(c).

• For second order RM codes, after the projection step, the RPA algorithm make use of both the decoding results of the projected codewords $\hat { y } _ { / \mathbb { B } _ { 1 } } , \hat { y } _ { / \mathbb { B } _ { 2 } } , \dotsc , \hat { y } _ { / \mathbb { B } _ { n - 1 } }$ and ˆ ˆ ˆthe original received word y to obtain the final decoding results while the algorithms in [11]–[13] only make use of $\hat { y } _ { / \mathbb { B } _ { 1 } } , \hat { y } _ { / \mathbb { B } _ { 2 } } , \dots , \hat { y } _ { / \mathbb { B } _ { n - 1 } }$ to obtain the coefficients of all ˆ ˆ ˆthe degree-2 monomials8 in the final decoding results. As discussed above, the projected codewords are more noisy than the original received words y. As a consequence, the performance of the algorithms in [11]–[13] is worse than that of the RPA algorithm; see Fig. 4(a),(b).

• The RPA algorithm uses $\hat { y } _ { / \mathbb { B } _ { 1 } } , \hat { y } _ { / \mathbb { B } _ { 2 } } , \hdots , \hat { y } _ { / \mathbb { B } _ { n } }$ together ˆ ˆ ˆwith the original received word y to correct errors bitwise in the original received word y while the algorithms in [11]–[13] use $\hat { y } _ { / \mathbb { B } _ { 1 } } , \hat { y } _ { / \mathbb { B } _ { 2 } } , \dots , \hat { y } _ { / \mathbb { B } _ { n - 1 } }$ to correct errors ˆ ˆwordwise among themselves.

In Fig. 4, we compare the RPA algorithm with the algorithms in [11]–[13] for decoding Reed-Muller codes over AWGN and BSC channels. Note that there are two parameters s and h in the Sidelnikov-Pershakov algorithm, where s is the list size of decoding each projected codeword, and h is the number of iterations when decoding the projected codewords. In our simulations, we set s   and h   since larger values = 4 = 3of s and h will not further improve the performance.

![](images/a0f506a18002ecf0492d6228ce9c3bc4313b4803e69fbb974097c0a06a4bd14e.jpg)  
(a) RM(9,2) over AWGN

![](images/f4b20e4e758de5706a23e6d1961f13423597c9a3206724956fe0f60b8a64bcd5.jpg)  
(b) RM(9,2) over BSC

![](images/49debeb33ea3e68f1df63c92ad274430dce83f17cb868d844da4a18fc0dbd11a.jpg)  
(c) RM(8,3) over BSC

Fig. 4. Comparison between the RPA algorithm and the algorithms in [11]–[13] for decoding Reed-Muller codes over AWGN and BSC channels. The curve with legend “Sakkour" is the performance of the algorithm in [12], [13], and the curves with legend “Sidelnikov-Pershakov" represent the performance of the algorithms in [11].  
![](images/e3c4087d770a56cce9f914876b935a9bb074c75ad268eaf19f637f13a86e8706.jpg)  
(a) RM(8,2)

![](images/486d7e736b9fffaf7ade8b36d999c71a926af0219bb1817082951f885f5947e4.jpg)  
(b) RM(9,3)  
Fig. 5. Comparison between the RPA decoding algorithm without list and Dumer’s recursive list decoding algorithm (the algorithm described in Section III of [16]) for decoding Reed-Muller codes over AWGN channels.

Next we compare the RPA algorithm with Dumer’s recursive list decoding algorithm [14]–[16]. Dumer’s list decoding algorithm provides a tradeoff between the decoding error probability and the decoding time. More precisely, if we set the list size to be large enough (e.g., exponential in n), then we can achieve the same performance as the maximal likelihood decoder, but we will also need exponential running time. If we choose small list size, then the algorithm runs fast but the decoding error will deteriorate.

In our simulations, we use the RPA algorithm and Dumer’s algorithm to decode RM codes over AWGN channels, and we find that the decoding error probability of RPA is slightly better (smaller) than Dumer’s algorithm, but the running time of RPA is typically larger. We have tested two cases RM , and RM , , and the performance is given in Fig. 5. For (9 3)RM , , the running time of our algorithm is 4.3ms, and the (8 2)running time of Dumer’s algorithm is 0.85ms. For RM , , (9 3)the running time of our algorithm is 3s, and the running time of Dumer’s algorithm is 0.14s.

In [19], simulation results are presented for RM , . (7 3)Their results are based on applying belief propagation to all minimum weight parity checks. This does seem indirectly related to using all first-order RM subcodes to decode. For RM , , the decoding complexities of these two approaches (7 3)are also similar. For RPA, each of  ∗  projections takes 127 63roughly ∗ operations to decode, giving 1.2M operations per 32 5iteration. For the algorithm in [19], there are 94448 minimum weight parity checks of weight 16 giving roughly 1.5M operations per iteration. It turns out that for $\mathcal { R M } ( 7 , 3 )$ , both (7 3)the performance and the running time of RPA decoder are similar to the algorithm in [19].

We also note that in [28], an algorithm with near-ML performance was also provided for $\mathcal { R M } ( 7 , 3 )$

## C. Parallelization and Acceleration

Another important advantage of the new decoding algorithm for RM codes over the SCL decoder for polar codes is that our algorithm naturally allows parallel implementation while the SCL decoder is not parallelizable. The key step in our algorithm for decoding a codeword of RM r, m is to decode the quotient space codes which are in $\mathbf { R M } ( r \mathrm { ~ - ~ } 1 , m \mathrm { ~ - ~ } 1 )$ ( 1 1)codes, and each of these can be decoded in parallel. Such a parallel structure is crucial to achieving high throughput and low latency.

Another way to accelerate the algorithm is to use only certain “voting $\mathrm { s e t s } " \colon$ In the projection step, we can take a subset of one-dimensional subspaces instead of all the one-dimensional subspaces. Then we still use recursive decoding followed by the aggregation step. In this way, we decode fewer $\mathbf { R M } ( r - 1 , m - 1 )$ codes, and if the voting sets were ( 1 1)chosen properly, we would obtain a similar decoding error probability with shorter running time. Note that in Section V we already gave a concrete choice of voting set in Algorithm 6, which indeed accelerates the decoding of high-rate RM codes with nearly-ML decoding error probability. At the same time, there might be other good voting sets to explore.

## D. Comparison With the Meta Converse Bound for Optimal Codes [29], [30]

We compared with upper bound from Corollary 39 and lower bound from Theorem 40 in [30]. More precisely, we provide the target error probability, the noise parameter of the channel, and the code dimension, then Corollary 39 and Theorem 40 in [30] give us upper and lower bound on the (optimal) code length. We found that RM ,  is nearly (8 2)optimal in terms of code length in the sense that the lower bound of code length given by [30, Theorem 40] is 251, which differs from the actual code length of RM codes by only 5. Then ${ \mathcal { R M } } ( 9 , 2 )$ is also close to optimal, where the lower (9 2)bound on code length is 500. However, for RM codes with larger order (dimension) and larger code length, the lower bound differs from the actual code length by at least , e.g., for ${ \mathcal { R M } } ( 9 , 3 )$ , the lower bound becomes 464.

## E. Optimal Scaling and Sharp Threshold of Reed-Muller Codes Over BSC Channels

Recently, Hassani et al. gave theoretical results backing the conjecture that RM codes have an almost optimal scaling-law over BSC channels under ML decoding [23], where optimal scaling-law means that for a fixed linear code, the decoding error probability of ML decoder transitions from  to  as a 0 1function of the crossover probability of the BSC channel in the sharpest manner (i.e., comparable to random codes). In particular, this implies that RM codes have sharper transition than polar codes under ML decoding (if capacity achieving). In this section we give simulation results that show that for BSC channels, Reed-Muller codes under the RPA decoder also have sharper transition than polar codes under SCL+list decoder.

In Figure 6, we plot the decoding error probability of RM codes and polar codes over BSC channels as a function of the channel crossover probability, where for RM codes we use the RPA decoder in Algorithm 1, and for polar codes we use SCL decoder with list size . We can see that in all  cases, 32 4the transition in the curve of RM codes is sharper than the transition in the curve of polar codes. To further quantify the transition width, we introduce the following common notation: Let us denote the channel crossover probability as . For a given code and a corresponding decoding algorithm, we write its decoding error probability over $\mathsf { B S C } ( \epsilon )$ as $P _ { e } ( \epsilon )$ . For $0 < \delta < 1 / 2$ ( ), we define the transition width9

$$
w ( \delta ) : = P _ { e } ^ { - 1 } ( 1 - \delta ) - P _ { e } ^ { - 1 } ( \delta ) .
$$

Clearly, w δ is a decreasing function. For a fixed value of δ, smaller $w ( \delta )$ means sharper transition and better scaling of the code and the corresponding decoder.

In Figure 7, we compare w . and w . between RM (0 1) (0 01)codes and polar codes with the same parameters, where we use the same decoders as above. We can see that RM codes always have smaller transition width than polar codes. Moreover, within the same code family, the transition width w . and (0 1)w . both decrease with the code length, meaning that (0 01)the transition becomes sharper as the code length increases. This phenomena has already been proved for ML decoders in [31] and [23].

## VII. EXTENSIONS

Here we mention a few possible extensions of the decoding algorithms.

1. The “voting sets" idea to further accelerate the RPA decoding, as employed in Section V and discussed in Section VI-C.

2. Our new algorithms make use of one-dimensional subspace reduction. In practice, we can change the $\mathbb { B } _ { 1 } , \dots , \mathbb { B } _ { n - 1 }$ in the RPA decoding algorithms to any of the s-dimensional subspaces, with different combinations possible. Note that in Section V, we already made use of this idea, where we chose $s = 2 .$

= 23. The RPA decoding algorithms can also be used to decode other codes that are supported on a vector space, or any code that has a well-defined notion of “code projection” that can be iteratively applied to produce eventually a trivial code (that can be decoded efficiently). In the case of RM codes, the quotient space projection has the specificity of producing again RM codes, and the trivial code is the Hadamard code that can be decoded using the FHT.

4. As discussed in Section III-A, we can use spectral decompositions or other relaxations in the Aggregation step instead of the majority voting, and depending on the regimes, one may take multiple iteration of the power-iteration method.

![](images/66209bbe648d54b25621de5a80e9376ff3fa55dd9d26bff5182dedd10fdbafd0.jpg)  
(a)RM(8,2) vs Polar codes with the same parameters

![](images/ca3c6a0214460cbdb57bde9d5ddf11a23344b9bc40e8d62b6dc65badd0988831.jpg)

![](images/130d9f33f62189895281d1b8a078f4e1dcd64036026f4c6c1f5beb1e3e39f813.jpg)  
（c） ${ \mathcal { R M } } ( 9 , 2 )$ vs Polar codes with the same parameters

(b) ${ \mathcal { R M } } ( 8 , 3 )$ vs Polar codes with the same parameters  
![](images/a6843c903719cb1effdfe8e4700bac0f992d871094675fd21d5fc3f3d4bc0197.jpg)  
(d) $\mathcal { R M } ( 1 0 , 2 )$ vs Polar codes with the same parameters

Fig. 6. Decoding error probability over BSC channels as a function of the channel crossover probability.  
![](images/48cdd85005f70948a4ab82b34dd575be4d8e718309dc9e8bdc7103321d8ea477.jpg)

![](images/4978cafa3e1dc2f3c5b51218b9ea5cdbcd54f9432fe388bdd3994c2d3c4b6b59.jpg)  
Fig. 7. Comparison of transition width w(0.1) and w(0.01) between different codes. $R ( m , r )$ refers to Reed-Muller codes, and $P ( m , r )$ refers to polar codes with the same length and dimension as $R ( m , r )$

## APPENDIX A

## PROOF OF LEMMA 1

Let $\pmb { b } _ { 1 } , \pmb { b } _ { 2 } , \dots , \pmb { b } _ { m }$ be a basis of E over $\mathbb { F } _ { 2 }$ such that the first s vectors $b _ { 1 } , { b _ { 2 } } , \ldots , { b _ { s } }$ form a basis of B. Let $e _ { 1 } , e _ { 2 } , \ldots , e _ { m }$ be the standard basis of $\mathbb { E } , \mathrm { i . e . }$ , all but the i-th coordinate of

$e _ { i }$ are . Then there is an $m \times m$ invertible matrix M such that

$$
( \pmb { b } _ { 1 } , \pmb { b } _ { 2 } , \dotsc , \pmb { b } _ { m } ) ^ { T } = M ( \pmb { e } _ { 1 } , \pmb { e } _ { 2 } , \dotsc , \pmb { e } _ { m } ) ^ { T } .
$$

Let $( z _ { 1 } , z _ { 2 } , \ldots , z _ { m } )$ be the coordinates of a point in E under (the standard basis $( e _ { 1 } , e _ { 2 } , \ldots , e _ { m } )$ , and let $( z _ { 1 } ^ { \prime } , z _ { 2 } ^ { \prime } , \ldots , z _ { m } ^ { \prime } )$

be the coordinates of the same point under the basis $\left( b _ { 1 } , b _ { 2 } , \ldots , b _ { m } \right)$ . Then

$$
( z _ { 1 } ^ { \prime } , z _ { 2 } ^ { \prime } , \ldots , z _ { m } ^ { \prime } ) = ( z _ { 1 } , z _ { 2 } , \ldots , z _ { m } ) { \cal M } ^ { - 1 } .
$$

Notice that $\mathbb { B } = \{ z : ( z _ { 1 } ^ { \prime } , z _ { 2 } ^ { \prime } , \ldots , z _ { s } ^ { \prime } ) \in \mathbb { F } _ { 2 } ^ { s } , z _ { s + 1 } ^ { \prime } = z _ { s + 2 } ^ { \prime } =$ $\cdots = z _ { m } ^ { \prime } = 0 \}$ = : ( ). Therefore for every coset $T \in \operatorname { \mathbb { E } } / \mathbb { B } ,$ =, the last $m \ : - \ : s$ = 0coordinates under the basis $\left( b _ { 1 } , b _ { 2 } , \ldots , b _ { m } \right)$ are the ( )same for all the points in T . As a result, we can use binary vectors of length $m - s$ to label the cosets, i.e.,

$$
\begin{array} { r l } & { [ a _ { 1 } , a _ { 2 } , \dotsc , a _ { m - s } ] : = } \\ & { \qquad \{ z : ( z _ { 1 } ^ { \prime } , z _ { 2 } ^ { \prime } , \dotsc , z _ { s } ^ { \prime } ) \in \mathbb { F } _ { 2 } ^ { s } , } \\ & { \qquad z _ { s + 1 } ^ { \prime } = a _ { 1 } , z _ { s + 2 } ^ { \prime } = a _ { 2 } , \dotsc , z _ { m } ^ { \prime } = a _ { m - s } \} . } \end{array}
$$

Next we associate every subset $A \subseteq [ m ]$ with another row vector $\nu _ { m } ^ { \prime } ( A )$ of length $2 ^ { m }$ [ ], whose components are indexed by $z = ( z _ { 1 } , z _ { 2 } , \ldots , z _ { m } ) \in \mathbb { E }$ . The vector $\nu _ { m } ^ { \prime } ( A )$ is defined as = (follows:

$$
{ \nu } _ { m } ^ { \prime } ( A , z ) = \prod _ { i \in A } z _ { i } ^ { \prime } ,
$$

where $\nu _ { m } ^ { \prime } ( A , z )$ is the component of $\nu _ { m } ^ { \prime } ( A )$ indexed by z, $\mathrm { i } . \mathrm { e } . , \nu _ { m } ^ { \prime } ( A , z )$ ) ( )is the evaluation of the polynomial $\prod _ { i \in A } Z _ { i } ^ { \prime }$ at z, where $\left( Z _ { 1 } ^ { \prime } , Z _ { 2 } ^ { \prime } , \ldots , Z _ { m } ^ { \prime } \right) = ( Z _ { 1 } , Z _ { 2 } , \ldots , Z _ { m } ) \bar { { \cal M } ^ { - 1 } } .$ . Since all ( ) = ( )the invertible linear transforms belong to the automorphism group of Reed-Muller codes [10], we have the following alternative characterization of RM codes

$$
\mathcal R \mathcal M ( m , r ) : = \Big \{ \sum _ { A \subseteq [ m ] , | A | \leq r } u ^ { \prime } ( A ) \pmb { \nu } _ { m } ^ { \prime } ( A ) : u ^ { \prime } ( A ) \in \{ 0 , 1 \}  \\  \mathrm { ~ f o r ~ a l l ~ } A \subseteq [ m ] , | A | \leq r \Big \} .
$$

It is easy to check that for every coset $T \quad = \quad$ $[ z _ { s + 1 } ^ { \prime } , z _ { s + 2 } ^ { \prime } , \ldots , z _ { m } ^ { \prime } ] \quad \in \quad \mathbb { E } / \mathbb { B } , \quad \mathrm { i f } \quad [ s ] \quad \subseteq \quad A _ { 1 }$ =then $\begin{array} { r l r } { \sum _ { z \in T } \pmb { \nu } _ { m } ^ { \prime } ( A , z ) } & { = } & { \prod _ { i \in ( A \backslash [ s ] ) } z _ { i } ^ { \prime } . } \end{array}$ [ ], and if $[ s ] \subsetneq A$ then $\begin{array} { r } { \sum _ { z \in T } \pmb { \nu } _ { m } ^ { \prime } ( A , z ) = 0 } \end{array}$ [ ]. Now let c be a codeword of $\mathcal { R M } ( m , r )$ , ( ) = 0then it can be written as $\begin{array} { r } { c = \sum _ { A \subset [ m ] , | A | < r } u ^ { \prime } ( A ) \pmb { \nu } _ { m } ^ { \prime } ( A ) } \end{array}$ ), and for every coset $T = [ z _ { s + 1 } ^ { \prime } , z _ { s + 2 } ^ { \prime } , \ldots , z _ { m } ^ { \prime } ] ^ { * } \in \mathbb { E } / \mathbb { B } .$ ) ( ), we have

$$
\begin{array} { c } { { \displaystyle \sum _ { z \in T } c ( z ) = \displaystyle \sum _ { A \supseteq [ s ] , | A | \leq r } u ^ { \prime } ( A ) \displaystyle \prod _ { i \in ( A \backslash [ s ] ) } z _ { i } ^ { \prime } } } \\ { { = \displaystyle \sum _ { A \subseteq ( [ m ] \backslash [ s ] ) , | A | \leq r - s } u ^ { \prime } ( A ) \displaystyle \prod _ { i \in A } z _ { i } ^ { \prime } . } } \end{array}
$$

Therefore every codeword in $\mathcal { Q } ( m , r , \mathbb { B } )$ corresponds to an $( m - s )$ -variate polynomial in $\mathbb { F } _ { 2 } [ Z _ { s + 1 } ^ { \prime } , Z _ { s + 2 } ^ { \prime } , \ldots , Z _ { m } ^ { \prime } ]$ with degree at most $r \mathrm { ~ - ~ } s .$ [ ], and this is exactly the definition of the $( r - s )$ -th order Reed-Muller code $\mathcal { R M } ( m - s , r - s )$

## APPENDIX B PROOF OF PROPOSITION 4

We need the following technical lemma to prove Proposition 4.

Lemma 3: Let $c _ { 0 } ~ = ~ ( c _ { 0 } ( z ) , z ~ \in ~ \mathbb { E } )$ be a codeword of $\mathcal { R M } ( m , r )$ . Let $L ^ { ( 1 ) } \ = \ ( L ^ { ( 1 ) } ( z ) , z \ \in \ \mathbb { E } )$ and $L ^ { ( 2 ) } ~ =$ $( L ^ { ( 2 ) } ( z ) , z \in \mathbb { E } )$ = ( ( ) )be two LLR vectors such that

$$
L ^ { ( 2 ) } ( z ) = ( - 1 ) ^ { c _ { 0 } ( z ) } L ^ { ( 1 ) } ( z ) \forall z \in \mathbb { E } .\tag{16}
$$

Denote $\begin{array} { r l r } { \hat { c } _ { 1 } } & { { } = } & { \mathrm { R P A \_ R M } ( L ^ { ( 1 ) } , m , r , N _ { \mathrm { m a x } } , \theta ) } \end{array}$ and $\begin{array} { r l } { \hat { c } _ { 2 } } & { { } = } \end{array}$ RPA\_RM $( L ^ { ( 2 ) } , m , r , N _ { \mathrm { m a x } } , \theta )$ . Then $\hat { c } _ { 1 } = \hat { c } _ { 2 } + c _ { 0 }$

Proof: We prove by induction on r. For the base case $r \ = \ 1$ , we use the ML decoder as described at the begin-= 1ning of this section. More precisely, according to $( 9 ) , \ \hat { c } _ { 2 } =$ $\mathtt { R P A \_ R M } ( L ^ { ( 2 ) } , m , 1 , N _ { \mathrm { m a x } } , \theta )$ is the codeword in $\mathcal { R M } ( m , 1 )$ (that maximizes

$$
\sum _ { z \in \mathbb { E } } \Big ( ( - 1 ) ^ { c ( z ) } L ^ { ( 2 ) } ( z ) \Big ) ,
$$

i.e., for all $c \in \mathcal { R M } ( m , 1 )$ , we have

$$
\sum _ { z \in \mathbb { E } } \Big ( ( - 1 ) ^ { \hat { c } _ { 2 } ( z ) } L ^ { ( 2 ) } ( z ) \Big ) \geq \sum _ { z \in \mathbb { E } } \Big ( ( - 1 ) ^ { c ( z ) } L ^ { ( 2 ) } ( z ) \Big ) .
$$

By (16), for all $c \in \mathcal { R M } ( m , 1 )$ , we have

$$
\sum _ { z \in \mathbb { E } } \Big ( ( - 1 ) ^ { \hat { c } _ { 2 } ( z ) \oplus c _ { 0 } ( z ) } L ^ { ( 1 ) } ( z ) \Big ) \geq \sum _ { z \in \mathbb { E } } \Big ( ( - 1 ) ^ { c ( z ) \oplus c _ { 0 } ( z ) } L ^ { ( 1 ) } ( z ) \Big ) .
$$

Since $c _ { 0 }$ is a codeword of $\mathcal { R M } ( m , 1 )$ , we have: $c _ { 0 } ~ +$ $\mathcal { R M } ( m , 1 ) = \mathcal { R M } ( m , 1 )$ ( 1). As a result, for all $c \in \mathcal { R M } ( m , 1 )$ (we have

$$
\sum _ { z \in \mathbb { E } } \Big ( ( - 1 ) ^ { \hat { c } _ { 2 } ( z ) \oplus c _ { 0 } ( z ) } L ^ { ( 1 ) } ( z ) \Big ) \geq \sum _ { z \in \mathbb { E } } \Big ( ( - 1 ) ^ { c ( z ) } L ^ { ( 1 ) } ( z ) \Big ) .
$$

Therefore, $\hat { c } _ { 2 } \oplus c _ { 0 }$ is the codeword in $\mathcal { R M } ( m , 1 )$ that maximizes

$$
\sum _ { z \in \mathbb { E } } \Big ( ( - 1 ) ^ { c ( z ) } L ^ { ( 1 ) } ( z ) \Big ) .
$$

Thus we conclude that $\hat { c } _ { 1 } = \hat { c } _ { 2 } \oplus c _ { 0 }$ . This establishes the base case.

For the inductive step, let us assume that the lemma holds for $r - 1$ and prove it for r. Notice that in Algorithm $3 , { \hat { c } } ( z )$ is simply determined by the sign of $L ( z )$ ˆ( ). It is easy to see that if ( )in Algorithm 4, the updated LLR vectors $\hat { L } ^ { ( 1 ) }$ and $\hat { L } ^ { ( 2 ) }$ always satisfy (16), then $\hat { c } _ { 1 } = \hat { c } _ { 2 } \oplus c _ { 0 }$ . Therefore, we only need to ˆ = ˆprove (16) for the updated LLR vectors $\hat { L } ^ { ( 1 ) }$ and $\hat { L } ^ { ( 2 ) }$

Assuming that $L ^ { \hat { ( } 1 ) }$ and $L ^ { ( 2 ) }$ satisfy (16), our task is to show that $\bar { \hat { L } } ^ { ( 2 ) } ( z ) = ( - 1 ) ^ { c _ { 0 } ( z ) } \hat { L } ^ { ( 1 ) } ( z )$ for all $z \in \mathbb { E } .$ . From the ( ) = ( 1) ( )analysis in Section IV, we know that

$$
\hat { L } ^ { ( i ) } ( z ) = \frac { 1 } { n - 1 } \sum _ { z ^ { \prime } \ne z } \alpha _ { i } ( z , z ^ { \prime } ) L ^ { ( i ) } ( z ^ { \prime } ) \mathrm { f o r } i = 1 , 2 .\tag{17}
$$

The coefficient $\alpha _ { i } ( z , z ^ { \prime } )$ is  if the decoding result of the corresponding $( r \mathrm { ~ - ~ } 1 ) { \bmod { } } { \bmod { } }$ 1 order RM code at the coset $\{ z , z ^ { \prime } \}$ is , and $\alpha _ { i } ( z , z ^ { \prime } )$ 1)is − if the decoding result at the coset $\{ z , z ^ { \prime } \}$ ( ) 1is  (see line  of Algorithm 4).

1 3Next we will show that $\bar { \alpha _ { 2 } ( z , z ^ { \prime } ) } = ( - 1 ) ^ { c _ { 0 } ( z ) \oplus c _ { 0 } ( z ^ { \prime } ) } \alpha _ { 1 } ( z , z ^ { \prime } )$ Note that $\alpha _ { i } ( z , z ^ { \prime } )$ is determined by the decoding result $\hat { y } _ { / \mathbb { B } } ^ { ( i ) } =$ RPA\_ $\mathbb { R M } ( L _ { / \mathbb { B } } ^ { ( i ) } , m - 1 , r - 1 , N _ { \operatorname* { m a x } } , \theta )$ , where $\mathbb { B } = \{ 0 , z \oplus z ^ { \prime } \}$

By (13),

$$
\begin{array} { r l } & { \Psi ( 1 , \mathfrak { d } , \mathfrak { p } , \Psi , \mathfrak { m } ) \quad \ : \forall \ : \ : \pi : : } \\ & { \ : \tilde { L } _ { \mathcal { F } _ { 2 } } ^ { ( 2 ) } ( T ) } \\ & { \ : = \ : \ln \left( \exp \left( \sum _ { \ell \in \mathcal { T } } L ^ { ( 3 ) } ( z ) \right) + 1 \right) - \ln \left( \displaystyle \sum _ { \ell \in \mathcal { T } } \exp \langle L ^ { ( 3 ) } ( z ) \rangle \right) } \\ & { \ : = \ln \left( \exp \left( \sum _ { \ell \in \mathcal { T } } \left( - 1 \right) ^ { \ell + 1 } ( z ) \right) + 1 \right) } \\ & { \ : \ : - \ln \left( \displaystyle \sum _ { \ell \in \mathcal { T } } \exp \left( ( - 1 ) ^ { \ell + 1 } ( z ) \right) \right) } \\ & { \ : \ : \ : \ : \ : \ : \ : \ : } \\ & { \ : \ : \ : - \ln \left( \displaystyle \sum _ { \ell \in \mathcal { T } } \exp \left( ( - 1 ) ^ { \ell + 1 } ( z ) \right) \right) } \\ & { \ : = \ : ( - 1 ) ^ { \Phi _ { \mathfrak { m e } } \times \sigma ( \tilde { L } ) } \left( \ln \left( \exp \left( \sum _ { \ell \in \mathcal { T } } L ^ { ( 3 ) } ( z ) \right) + 1 \right) \right) } \\ & { \ : \ : \ : \ : \ : \ : \ : \ : \ : } \\ & { \ : \ : \ : \ : \ : - \ln \left( \displaystyle \sum _ { \ell \in \mathcal { T } } \exp \langle L ^ { ( 3 ) } ( z ) \rangle \right) } \\ & { \ : = \ : ( - 1 ) ^ { \Omega _ { \mathfrak { m e } } \times \sigma ( \tilde { L } ) } \frac { 1 } { L ^ { ( 3 ) } ( T ) } ( T ) , } \end{array}
$$

Let us write $c _ { 0 } ( T ) \ : = \ \bigoplus _ { z \in T } c _ { 0 } ( z )$ . Then $L _ { / \mathbb { B } } ^ { ( 2 ) } ( T ) =$ $( - 1 ) ^ { c _ { 0 } ( T ) } L _ { / \mathbb { B } } ^ { ( 1 ) } ( T )$ for all $T \in \mathbb { E } / \mathbb { B }$ . Moreover, since $c _ { 0 }$ is a ( 1)codeword of $\mathcal { R M } ( m , r )$ and B is a one-dimensional subspace of $\mathbb { E } ,$ ( ) by Lemma 1 we know that $( c _ { 0 } ( T ) , T \ \in \ \mathbb { E } / \mathbb { B } )$ is a codeword of $\mathcal { R M } ( m - 1 , r - 1 )$ ( ( ) ). Therefore, the codeword $( c _ { 0 } ( T ) , T \in \mathbb { E } / \mathbb { B } )$ and the two LLR vectors $( L _ { / \mathbb { B } } ^ { ( 1 ) } ( T ) , T \in$ $\mathbb { E } / \mathbb { B } )$ and $( L _ { / \mathbb { B } } ^ { ( 2 ) } ( T ) , T \in \mathbb { E } / \mathbb { B } )$ satisfy the conditions of this lemma. By the induction hypothesis, $\hat { y } _ { / \mathbb { B } } ^ { ( 2 ) } ( T ) = \hat { y } _ { / \mathbb { B } } ^ { ( 1 ) } ( T ) \oplus$ $c _ { 0 } ( T )$ for all $T \in \mathbb { E } / \mathbb { B }$ ˆ ( ) =. As a result, we have $\alpha _ { 2 } ( z , z ^ { \prime } ) ~ =$ $( - 1 ) ^ { \dot { c } _ { 0 } ( z ) \oplus c _ { 0 } ( z ^ { \prime } ) } \alpha _ { 1 } ( z , z ^ { \prime } )$ ( ) =. Taking this into (17), we conclude ( 1)that for all $z \in \mathbb { E } .$

$$
\begin{array} { l } { { \displaystyle \hat { L } ^ { ( 2 ) } ( z ) } } \\ { { \displaystyle = \frac { 1 } { n - 1 } \sum _ { z ^ { \prime } \neq z } \alpha _ { 2 } ( z , z ^ { \prime } ) L ^ { ( 2 ) } ( z ^ { \prime } ) } } \\ { { \displaystyle = \frac { 1 } { n - 1 } \sum _ { z ^ { \prime } \neq z } \Big ( ( - 1 ) ^ { c _ { 0 } ( z ) \oplus c _ { 0 } ( z ^ { \prime } ) } \alpha _ { 1 } ( z , z ^ { \prime } ) ( - 1 ) ^ { c _ { 0 } ( z ^ { \prime } ) } L ^ { ( 1 ) } ( z ^ { \prime } ) \Big ) } } \\ { { \displaystyle = ( - 1 ) ^ { c _ { 0 } ( z ) } \frac { 1 } { n - 1 } \sum _ { z ^ { \prime } \neq z } \alpha _ { 1 } ( z , z ^ { \prime } ) L ^ { ( 1 ) } ( z ^ { \prime } ) = ( - 1 ) ^ { c _ { 0 } ( z ) } \hat { L } ^ { ( 1 ) } ( z ) . } } \end{array}
$$

This completes the proof of the inductive step and establishes the lemma. □

Proof of Proposition 4: Since W is a BMS channel, there is a permutation π of the output alphabet W satisfying the two conditions in Definition 3. Since both $c _ { 1 }$ and $c _ { 2 }$ are codewords of $\mathcal { R M } ( m , r ) , c _ { 0 } : = c _ { 1 } + c _ { 2 }$ is also a codeword of $\mathcal { R M } ( m , r )$ ( ) := +. Clearly, both channel output vectors $Y _ { 1 }$ and $Y _ { 2 }$ (belong to $\mathcal { W } ^ { n }$ . Now we define a permutation $\pi ^ { c _ { 0 } }$ on $\mathcal { W } ^ { n }$ : For any $y = ( y ( z ) , z \in \mathbb { E } ) \in \mathcal { W } ^ { n }$

$$
\pi ^ { c _ { 0 } } ( y ) : = ( \pi ^ { c _ { 0 } ( z ) } ( y ( z ) ) , z \in \mathbb { E } ) .
$$

Notice that $c _ { 0 } ( z )$ is either  or , and $\pi ^ { 0 }$ is the identity map. ( ) 0Since π is a permutation on $\mathcal { W } , \ \pi ^ { c _ { 0 } }$ is clearly a permutation on $\mathcal { W } ^ { n }$ . For a given $y = ( y ( z ) , z \in \mathbb { E } ) \in \mathcal { W } ^ { n }$ , we denote the = ( ( )LLR vector corresponding to y as $L _ { y } ^ { ( 1 ) } : = ( L _ { y } ^ { ( 1 ) } ( z ) , z \in \mathbb { E } )$ , i.e., $L _ { y } ^ { ( 1 ) } ( z ) = \mathrm { L L R } ( y ( z ) )$ for all $z \in \mathbb { E } ,$ and we denote the ( ) = LLR( ( ))LLR vector corresponding to $\pi ^ { c _ { 0 } } ( y )$ as $L _ { y } ^ { ( 2 ) } : = ( L _ { y } ^ { ( 2 ) } ( z )$ $z \in \mathbb { E } )$ , i.e., $L _ { y } ^ { ( 2 ) } ( z ) \overset { \cdot } { = } \mathrm { L L R } ( \pi ^ { c _ { 0 } ( z ) } ( y ( z ) ) )$ for all $z \in \mathbb { E } .$ . By the ) ( ) = LLR( ( ( )))property of π (see Definition 3), we have

Algorithm 8 The RPA\_RM Decoding Function for BSC   
Input: The corrupted codeword $y = ( y ( z ) , z \in \{ 0 , 1 \} ^ { m } ) ;$ the   
= ( ( )parameters of the Reed-Muller code m and $r ;$ 0 1 ) the maximal   
number of iterations $N _ { \mathrm { m a x } }$   
Output: The decoded codeword c   
1: for $i = 1 , 2 , \ldots , N _ { \mathrm { m a x } }$ do   
2: = 1 2Initialize changevote $\mathopen : ( z ) , z \in \{ 0 , 1 \} ^ { m } )$ as an all-zero   
(vector indexed by $z \in \{ 0 , 1 \} ^ { m }$   
3: for each non-zero $z _ { 0 } \in \{ 0 , 1 \} ^ { m }$ do   
4: Set $\mathbb { B } = \{ 0 , z _ { 0 } \}$   
5: $\hat { y } _ { / \mathbb { B } } \gets \mathtt { R P A \_ R M } ( y _ { / \mathbb { B } } , m - 1 , r - 1 , N _ { \operatorname* { m a x } } )$   
6: $\triangleright \operatorname { I f } r = \dot { 2 } ,$ 1 1 ) then we use the Fast Hadamard   
= 2Transform to decode the first-order RM code [10]   
7: for each $z \in \{ 0 , 1 \} ^ { m }$ do   
8: if $y _ { / \mathbb { B } } ( [ z + \mathbb { B } ] ) \neq \hat { y } _ { / \mathbb { B } } ( [ z + \mathbb { B } ] )$ then   
9: ([ + ]) = ˆ ([ + ])changevote z ← changevote $\frac { 3 } { 2 } ( z ) + 1 0$   
( )Here addition is between real numbers   
10: end if   
11: end for   
12: end for   
13: numofchange $ 0$   
14: $n \gets 2 ^ { m }$   
15: 2for each $z \in \{ 0 , 1 \} ^ { m }$ do   
16: 0 1if changevot $\textstyle \operatorname { e } ( z ) > { \frac { n - 1 } { 2 } }$ then   
17: $y ( z )  y ( z ) \oplus 1 \qquad { \mathit { \Delta } } [ { \mathit { \Delta } } ]$ Here addition is over $\mathbb { F } _ { 2 }$   
18: numofchange ← numofchange  - Here   
addition is between real numbers   
19: end if   
20: end for   
21: if numofchange  then   
22: break = 0- Exit the first for loop of this function   
23: end if   
24: end for   
25: $\hat { c } \gets y$   
ˆ26: return c

$$
L _ { y } ^ { ( 2 ) } ( z ) = ( - 1 ) ^ { c _ { 0 } ( z ) } L _ { y } ^ { ( 1 ) } ( z ) \forall z \in \mathbb { E } .
$$

Since $c _ { 0 } \in \mathcal { R M } ( m , r )$ , by Lemma 3 we know that

$$
\begin{array} { r l } & { \mathrm { R P \mathbb A \_ R M } ( L _ { y } ^ { ( 1 ) } , m , r , N _ { \mathrm { m a x } } , \theta ) } \\ & { = \mathrm { R P \mathbb A \_ R M } ( L _ { y } ^ { ( 2 ) } , m , r , N _ { \mathrm { m a x } } , \theta ) + c _ { 0 } . } \end{array}
$$

As a result, RPA\_RM $\mathrm { \ i } ( L _ { y } ^ { ( 1 ) } , m , r , N _ { \mathrm { m a x } } , \theta ) \neq { { c } _ { 1 } }$ if and only if RPA\_RM $( L _ { y } ^ { ( 2 ) } , m , r , N _ { \mathrm { m a x } } , \theta ) \neq { c _ { 2 } } .$

(For a vector $y \in \mathcal { W } ^ { n }$ ) =and a codeword $c \in \mathcal { R M } ( m , r )$ we use $W ^ { n } ( y | c )$ ( )to denote the probability of outputting y when ( )the transmitted codeword is c. Again by the property of $\pi ,$ it is easy to see that

$$
W ^ { n } ( y | c _ { 1 } ) = W ^ { n } ( \pi ^ { c _ { 0 } } ( y ) | c _ { 2 } ) \forall y \in \mathcal { W } ^ { n } .
$$

Algorithm 9 The RPA\_RM Decoding Function for General   
Binary-Input Memoryless Channels   
Input: The LLR vector $( L ( z ) , z \in \{ 0 , 1 \} ^ { m } )$ ; the parameters   
( ( ) 0 1 )of the Reed-Muller code m and r; the maximal number of   
iterations $N _ { \mathrm { m a x } } ;$ the exiting threshold θ   
Output: The decoded codeword $\hat { c } = ( \hat { c } ( z ) , z \in \{ 0 , 1 \} ^ { m } )$   
1: E  { , }m   
:2: for $i = 1 , 2 , \ldots , N _ { \mathrm { m a x } }$ do   
= 1 23: Initialize cumuLLR z , z ∈ E as an all-zero vector   
indexed by $z \in \mathbb { E }$   
4: for each non-zero $z _ { 0 } \in \mathbb { E }$ do   
5: Set $\mathbb { B } = \{ 0 , z _ { 0 } \}$   
6: $L _ { / \mathbb { B } } \gets ( L _ { / \mathbb { B } } ( T ) , T \in \mathbb { E } / \mathbb { B } ) \textrm { \textsf { \textsf { P } } } L _ { / \mathbb { B } } ( T )$ is calculated   
from $( { \dot { L } } ( z ) , z \in { \dot { \mathbb { E } } } )$ ( ) )according to (13)   
7: $\boldsymbol { \hat { y } } _ { / \mathbb { B } } \gets \mathrm { R P A \_ R M } ( L _ { / \mathbb { B } } , m - 1 , r - 1 , N _ { \operatorname* { m a x } } , \theta )$   
8: $\triangleright \operatorname { I f } r = 2 ,$ 1 1 ) then we use the Fast Hadamard   
= 2Transform to decode the first-order RM code   
9: for each $z \in \mathbb { E }$ do   
10: if $\hat { y } _ { / \mathbb { B } } ( [ z + \mathbb { B } ] ) = 0$ then   
11: ˆ ([ + ]) = 0cumuLLR z ← cumuLLR $( z ) + L ( z \oplus z _ { 0 } )$   
12: else - $\hat { y } _ { / \mathbb { B } }$ ( ) + ( )is the decoded codeword,   
so $\hat { y } _ { / \mathbb { B } } ( [ z + \mathbb { B } ] )$ ˆis either  or   
13: ]) 0 1cumuLLR z ← cumuLLR z − L z ⊕ z0   
14: end if   
15: end for   
16: end for   
17: numofchange $ 0$   
18: $n \gets 2 ^ { m }$   
19: 2for each $z \in \mathbb { E }$ do   
20: cumuLLR z ← cumuLLR(z)   
21: if $| \mathrm { c u m u L L R } ( z ) - \tilde { L } ^ { \smash { n - 1 } } ( \bar { z } ) | > \theta | L ( z ) |$ then   
22: ( ) ( ) ( )numofchange ← numofchange   
23: + 1- Here addition is between real numbers   
24: end if   
25: $L ( z ) \gets$ cumuLLR z   
26: ( )end for   
27: if numofchange   then   
28: break = 0- Exit the first for loop of this function   
29: end if   
30: end for   
31: for each $z \in \mathbb { E }$ do   
32: if $L ( z ) > 0$ then   
33: ( ) 0c z ←   
34: ˆ(else   
35: c z ←   
36: ˆ( )end if   
37: end for   
38: return c

Recall that in Proposition 4, we use $L ^ { ( 1 ) }$ and $L ^ { ( 2 ) }$ to denote the random LLR vectors corresponding to the random channel outputs when transmitting $c _ { 1 }$ and $c _ { 2 } .$ , respectively. Therefore,

$$
\begin{array} { r l } & { \quad \mathbb { P } \big ( \mathtt { R P A \_ R M } ( L ^ { ( 1 ) } , m , r , N _ { \operatorname* { m a x } } , \theta ) \ne c _ { 1 } \big ) } \\ & { = \displaystyle \sum _ { y \in \mathcal { W } ^ { n } } W ^ { n } ( y | c _ { 1 } ) \mathbb { 1 } \big [ \mathtt { R P A \_ R M } ( L _ { y } ^ { ( 1 ) } , m , r , N _ { \operatorname* { m a x } } , \theta ) \ne c _ { 1 } \big ] } \end{array}
$$

$$
= \sum _ { y \in \mathcal { W } ^ { n } } W ^ { n } ( \pi ^ { c _ { 0 } } ( y ) | c _ { 2 } ) \mathbb { 1 } \bigl [ \mathrm { R P } \mathbb { A } _ { - } \mathrm { R M } ( L _ { y } ^ { ( 2 ) } , m , r , N _ { \operatorname* { m a x } } , \theta ) \neq c _ { 2 } \bigr ]
$$

$$
\begin{array} { r l } & { = \mathbb { P } \big ( \mathtt { R P A \_ R M } ( L ^ { ( 2 ) } , m , r , N _ { \mathrm { m a x } } , \theta ) \ne c _ { 2 } \big ) . } \end{array}
$$

This completes the proof of Proposition 4.

## APPENDIX C

See Algorithms 8 and 9.

## ACKNOWLEDGMENT

The authors would like to thank Alexander Barg and Ilya Dumer for pointing out several references and giving useful feedback. They also thank Kirill Ivanov for useful discussions and feedback.

## REFERENCES

[1] M. Ye and E. Abbe, “Recursive projection-aggregation decoding of Reed–Muller codes,” in Proc. IEEE Int. Symp. Inf. Theory (ISIT), Jul. 2019, pp. 2064–2068.

[2] I. Reed, “A class of multiple-error-correcting codes and the decoding scheme,” Trans. IRE Prof. Group Inf. Theory, vol. 4, no. 4, pp. 38–49, Sep. 1954.

[3] E. Arikan, “Channel polarization: A method for constructing capacityachieving codes for symmetric binary-input memoryless channels,” IEEE Trans. Inf. Theory, vol. 55, no. 7, pp. 3051–3073, Jul. 2009.

[4] E. Arkan, “A performance comparison of polar codes and Reed–Muller codes,” IEEE Commun. Lett., vol. 12, no. 6, pp. 447–449, Jun. 2008.

[5] M. Mondelli, S. H. Hassani, and R. L. Urbanke, “From polar to Reed–Muller codes: A technique to improve the finite-length performance,” IEEE Trans. Commun., vol. 62, no. 9, pp. 3084–3091, Sep. 2014.

[6] S. Kudekar, S. Kumar, M. Mondelli, H. D. Pfister, E. ¸Sa¸so ˇglu, and R. L. Urbanke, “Reed–Muller codes achieve capacity on erasure channels,” IEEE Trans. Inf. Theory, vol. 63, no. 7, pp. 4298–4316, Jul. 2017.

[7] E. Abbe, A. Shpilka, and A. Wigderson, “Reed–Muller codes for random erasures and errors,” IEEE Trans. Inf. Theory, vol. 61, no. 10, pp. 5229–5252, Oct. 2015.

[8] E. Abbe and M. Ye, “Reed–Muller codes polarize,” in Proc. IEEE 60th Annu. Symp. Found. Comput. Sci. (FOCS), Nov. 2019, pp. 273–286.

[9] E. Abbe, A. Shpilka, and M. Ye, “Reed–Muller codes: Theory and algorithms,” 2020, arXiv:2002.03317. [Online]. Available: http://arxiv.org/abs/2002.03317

[10] F. J. MacWilliams and N. J. A. Sloane, The Theory of Error-Correcting Codes. Amsterdam, The Netherlands: Elsevier, 1977.

[11] V. M. Sidel’nikov and A. S. Pershakov, “Decoding of Reed–Muller codes with a large number of errors,” Probl. Peredachi Inf., vol. 28, no. 3, pp. 80–94, 1992.

[12] P. Loidreau and B. Sakkour, “Modified version of Sidel’nikov-Pershakov decoding algorithm for binary second order Reed–Muller codes,” in Proc. 9th Int. Workshop Algebraic Combinat. Coding Theory (ACCT), Kranevo, Bulgaria, 2004, pp. 266–271.

[13] B. Sakkour, “Decoding of second order Reed–Muller codes with a large number of errors,” in Proc. IEEE Inf. Theory Workshop, Sep. 2005, pp. 176–178.

[14] I. Dumer, “Recursive decoding and its performance for low-rate Reed–Muller codes,” IEEE Trans. Inf. Theory, vol. 50, no. 5, pp. 811–823, May 2004.

[15] I. Dumer, “Soft-decision decoding of Reed–Muller codes: A simplified algorithm,” IEEE Trans. Inf. Theory, vol. 52, no. 3, pp. 954–963, Mar. 2006.

[16] I. Dumer and K. Shabunov, “Soft-decision decoding of Reed–Muller codes: Recursive lists,” IEEE Trans. Inf. Theory, vol. 52, no. 3, pp. 1260–1266, Mar. 2006.

[17] R. Saptharishi, A. Shpilka, and B. L. Volk, “Efficiently decoding Reed–Muller codes from random errors,” IEEE Trans. Inf. Theory, vol. 63, no. 4, pp. 1954–1960, Apr. 2016.

[18] O. Sberlo and A. Shpilka, “On the performance of Reed–Muller codes with respect to random errors and erasures,” Proc. 14th Annu. ACM-SIAM Symp. Discrete Algorithms, Dec. 2020, pp. 1357–1376.

[19] E. Santi, C. Hager, and H. D. Pfister, “Decoding Reed–Muller codes using minimum-weight parity checks,” in Proc. IEEE Int. Symp. Inf. Theory (ISIT), Jun. 2018, pp. 1296–1300.

[20] I. Tal and A. Vardy, “List decoding of polar codes,” IEEE Trans. Inf. Theory, vol. 61, no. 5, pp. 2213–2226, May 2015.

[21] D. Chase, “Class of algorithms for decoding block codes with channel measurement information,” IEEE Trans. Inf. Theory, vol. IT-18, no. 1, pp. 170–182, Jan. 1972.

[22] Final Report of 3 GPP TSG RAN WG1 #87 v1.0.0. Accessed: Feb. 8, 2017. [Online]. Available: http://www.3gpp.org/ftp/tsg\_ran/ WG1\_RL1/TSGR1\_87/Report/

[23] H. Hassani, S. Kudekar, O. Ordentlich, Y. Polyanskiy, and R. Urbanke, “Almost optimal scaling of Reed–Muller codes on BEC and BSC channels,” in Proc. IEEE Int. Symp. Inf. Theory (ISIT), Jun. 2018, pp. 311–315.

[24] R. R. Green, “A serial orthogonal decoder,” JPL Space Programs Summary, vol. IV, nos. 37–39, pp. 247–253, 1966.

[25] Y. Be’ery and J. Snyders, “Optimal soft decision block decoders based on fast Hadamard transform,” IEEE Trans. Inf. Theory, vol. IT-32, no. 3, pp. 355–364, May 1986.

[26] A. Balatsoukas-Stimming, M. Bastani Parizi, and A. Burg, “LLR-based successive cancellation list decoding of polar codes,” IEEE Trans. Signal Process., vol. 63, no. 19, pp. 5165–5179, Oct. 2015.

[27] G. Sarkis, P. Giard, A. Vardy, C. Thibeault, and W. J. Gross, “Fast list decoders for polar codes,” IEEE J. Sel. Areas Commun., vol. 34, no. 2, pp. 318–328, Feb. 2016.

[28] S. A. Hashemi, N. Doan, M. Mondelli, and W. J. Gross, “Decoding Reed–Muller and polar codes by successive factor graph permutations,” in Proc. IEEE 10th Int. Symp. Turbo Codes Iterative Inf. Process. (ISTC), Dec. 2018, pp. 1–5.

[29] Y. Polyanskiy, H. V. Poor, and S. Verdu, “Channel coding rate in the finite blocklength regime,” IEEE Trans. Inf. Theory, vol. 56, no. 5, pp. 2307–2359, May 2010.

[30] Y. Polyanskiy, Channel Coding: Non-Asymptotic Fundamental Limits. Princeton, NJ, USA: Princeton Univ., 2010.

[31] J.-P. Tillich and G. Zémor, “Discrete isoperimetric inequalities and the probability of a decoding error,” Combinat. Probab. Comput., vol. 9, no. 5, pp. 465–479, Sep. 2000.

Min Ye received the B.S. degree in electrical engineering from Peking University, Beijing, China, in 2012, and the Ph.D. degree from the Department of Electrical and Computer Engineering, University of Maryland, College Park, MA, USA, in 2017. He then spent two years as a Postdoctoral Researcher at Princeton University. Since 2019, he has been an Assistant Professor with the Data Science and Information Technology Research Center, Tsinghua-Berkeley Shenzhen Institute, Shenzhen, China. His research interests include coding theory, information theory, differential privacy, and machine learning. He received the 2017 IEEE Data Storage Best Paper Award.

Emmanuel Abbe received the Ph.D. degree from the Department of Electrical Engineering and Computer Science, Massachusetts Institute of Technology, in 2008, and the M.S. degree from the Department of Mathematics, Ecole Polytechnique Fédérale de Lausanne (EPFL), Switzerland, in 2003. In 2012, he joined Princeton University as an Assistant Professor and became an Associate Professor in 2016, jointly in the Program for Applied and Computational Mathematics and the Department of Electrical Engineering. He has been an Associate Faculty with the Department of Mathematics, Princeton University, since 2016. He is also a Professor with the Mathematics Institute, School of Computer and Communication Sciences, EPFL. He is a recipient of the Foundation Latsis International Prize, the Bell Labs Prize, the NSF CAREER Award, the Google Faculty Research Award, the Walter Curtis Johnson Prize, and the von Neumann Fellowship from the Institute for Advanced Study.