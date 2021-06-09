Automatic line-break for too long sentence

```latex
\usepackage{tabularx} % tabularx 
\begin{table}[t!]
    \centering
    \small 
    \begin{tabularx}{\linewidth}{X|c} % use X to denote where need auto break and specify the width 
    \toprule
      Input   &  Label \\
    \midrule
     \textbf{Text A}:  Organs are collections of tissues grouped together performing a common function.
  &  1  \\ 
   \midrule  
  \textbf{Text A}: Organs are collections of tissues grouped together performing a common function. 
  \textbf{Text B}: Text B: Does this sentence contains a definition? 
& 1 \\ 
    \bottomrule
    \end{tabularx}
    \caption{Caption}
    \label{tab:my_label}
\end{table}
```