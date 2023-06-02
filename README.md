# MI_ESTIMATION_BASE
This repository contains a base code for estimation of mutual information 

Example 1: MNIST dataset
Original label is the digit shown by a handwritten image
L(x) is defined as if the digit is even (label 0) or odd (label 1)
We find the estimation (a tight lower bound) of I(x,L(x)) using neural estimation
Theoretically, if we can identify the digit given an image, I(x,L(x))=H(L(x)). For a balanced dataset, H(L(x))=log(2)=0.6931
