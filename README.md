# loupe-pytorch

Learnable mOdUle for Pooling fEatures(LOUPE) implementation in pytorch and it's example use
Converted the code from Antoine Miech, Ivan Laptev, Josef Sivic presented in <https://github.com/antoine77340/LOUPE>
Inspired from <https://gist.github.com/catta202000/cbf5cae9199ccd72adddef452358df3b>

The code is not extensively tested, please ping me if you notice any problems.

LSTM-Loupe shows the use of NetVLAD layer for pooling the outputs of multiple LSTM's.

loupe-pytorch includes the pytorch implementations of multiple learnable pooling layers:
NetVLAD, NetRVLAD, SoftDBoW and NetFV
NetFV implementation is currently not working.
