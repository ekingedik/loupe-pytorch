""" Learnable mOdUle for Pooling fEatures (LOUPE)

Implementation of LOUPE of Antoine Miech, Ivan Laptev, Josef Sivic in pytorch.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PoolingBaseModel(nn.Module):
    """Inherit from this class when implementing new models."""

    def __init__(
        self,
        feature_size,
        max_samples,
        cluster_size,
        output_dim,
        gating=True,
        add_batch_norm=True,
        is_training=True,
    ):
        """Initialize a Learnable Pooling block.
        Args:
        feature_size: Dimensionality of the input features.
        max_samples: The maximum number of samples to pool.
        cluster_size: The number of clusters.
        output_dim: size of the output space after dimension reduction.
        add_batch_norm: (bool) if True, adds batch normalization.
        is_training: (bool) Whether or not the graph is training.
        """
        super(PoolingBaseModel, self).__init__()
        self.feature_size = feature_size
        self.max_samples = max_samples
        self.output_dim = output_dim
        self.is_training = is_training
        self.gating = gating
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size

    def forward(self, reshaped_input):
        raise NotImplementedError("Models should implement the forward pass.")


class GatingContext(nn.Module):
    def __init__(self, dim, add_batch_norm=True):
        super(GatingContext, self).__init__()
        self.dim = dim
        self.add_batch_norm = add_batch_norm
        self.gating_weights = nn.Parameter(
            torch.nn.init.normal_(
                torch.empty(dim, dim), mean=0, std=(1 / math.sqrt(dim)),
            )
        )
        self.sigmoid = nn.Sigmoid()
        if add_batch_norm:
            self.gating_biases = None
            self.batch_norm = nn.BatchNorm1d(dim)
        else:
            self.gating_biases = nn.Parameter(
                torch.nn.init.normal_(
                    torch.empty(dim), mean=0, std=(1 / math.sqrt(dim)),
                )
            )
            self.batch_norm = None

    def forward(self, x):
        gates = torch.matmul(x, self.gating_weights)
        if self.add_batch_norm:
            gates = self.batch_norm(gates)
        else:
            gates = gates + self.gating_biases
        gates = self.sigmoid(gates)
        activation = x * gates
        return activation


class NetVLAD(PoolingBaseModel):
    """Creates a NetVLAD class.
    """

    def __init__(
        self,
        feature_size,
        max_samples,
        cluster_size,
        output_dim,
        gating=True,
        add_batch_norm=True,
        is_training=True,
    ):
        super(NetVLAD, self).__init__(
            feature_size=feature_size,
            max_samples=max_samples,
            cluster_size=cluster_size,
            output_dim=output_dim,
            gating=gating,
            add_batch_norm=add_batch_norm,
            is_training=is_training,
        )

        self.softmax = nn.Softmax(dim=-1)
        self.cluster_weights = nn.Parameter(
            torch.nn.init.normal_(
                torch.empty(self.feature_size, self.cluster_size),
                mean=0,
                std=(1 / math.sqrt(self.feature_size)),
            )
        )

        self.cluster_weights2 = nn.Parameter(
            torch.nn.init.normal_(
                torch.empty(1, self.feature_size, self.cluster_size),
                mean=0,
                std=(1 / math.sqrt(self.feature_size)),
            )
        )
        self.hidden1_weights = nn.Parameter(
            torch.nn.init.normal_(
                torch.empty(self.cluster_size * self.feature_size, self.output_dim),
                mean=0,
                std=(1 / math.sqrt(self.cluster_size)),
            )
        )

        if add_batch_norm:
            self.cluster_biases = None
            self.batch_norm = nn.BatchNorm1d(self.cluster_size)
        else:
            self.cluster_biases = nn.Parameter(
                torch.nn.init.normal_(
                    torch.empty(self.cluster_size),
                    mean=0,
                    std=(1 / math.sqrt(self.feature_size)),
                )
            )
            self.batch_norm = None

        if gating:
            self.context_gating = GatingContext(
                self.output_dim, add_batch_norm=self.add_batch_norm
            )

    def forward(self, input):
        # Reshape the tensor to be 'batch_size*max_samples' x 'feature_size'
        reshaped_input = input.view(-1, self.feature_size)
        activation = torch.matmul(reshaped_input, self.cluster_weights)

        if self.add_batch_norm:
            activation = self.batch_norm(activation)
        else:
            activation += self.cluster_biases

        activation = self.softmax(activation)
        activation = activation.view(-1, self.max_samples, self.cluster_size)
        a_sum = activation.sum(-2, keepdim=True)
        a = a_sum * self.cluster_weights2

        activation = activation.permute(0, 2, 1)
        reshaped_input = reshaped_input.view(-1, self.max_samples, self.feature_size)

        vlad = torch.matmul(activation, reshaped_input)
        vlad = vlad.permute(0, 2, 1)
        vlad = vlad - a
        vlad = F.normalize(vlad, dim=1, p=2)
        vlad = vlad.contiguous().view((-1, self.cluster_size * self.feature_size))
        vlad = F.normalize(vlad, dim=1, p=2)
        vlad = torch.matmul(vlad, self.hidden1_weights)
        if self.gating:
            vlad = self.context_gating(vlad)

        return vlad


class NetRVLAD(PoolingBaseModel):
    """Creates a NetRVLAD class (Residual-less NetVLAD).
    """

    def __init__(
        self,
        feature_size,
        max_samples,
        cluster_size,
        output_dim,
        gating=True,
        add_batch_norm=True,
        is_training=True,
    ):
        super(NetRVLAD, self).__init__(
            feature_size=feature_size,
            max_samples=max_samples,
            cluster_size=cluster_size,
            output_dim=output_dim,
            gating=gating,
            add_batch_norm=add_batch_norm,
            is_training=is_training,
        )

        self.cluster_weights = nn.Parameter(
            torch.nn.init.normal_(
                torch.empty(self.feature_size, self.cluster_size),
                mean=0,
                std=(1 / math.sqrt(self.feature_size)),
            )
        )

        self.hidden1_weights = nn.Parameter(
            torch.nn.init.normal_(
                torch.empty(self.cluster_size * self.feature_size, self.output_dim),
                mean=0,
                std=(1 / math.sqrt(self.cluster_size)),
            )
        )

        self.softmax = nn.Softmax(dim=-1)

        if add_batch_norm:
            self.cluster_biases = None
            self.batch_norm = nn.BatchNorm1d(self.cluster_size)
        else:
            self.cluster_biases = nn.Parameter(
                torch.nn.init.normal_(
                    torch.empty(self.cluster_size),
                    mean=0,
                    std=(1 / math.sqrt(self.feature_size)),
                )
            )
            self.batch_norm = None

        if gating:
            self.context_gating = GatingContext(
                self.output_dim, add_batch_norm=self.add_batch_norm
            )

    def forward(self, input):
        """Forward pass of a NetRVLAD block.
        Args:
        reshaped_input: If your input is in that form:
        'batch_size' x 'max_samples' x 'feature_size'
        It should be reshaped in the following form:
        'batch_size*max_samples' x 'feature_size'
        by performing:
        reshaped_input = tf.reshape(input, [-1, features_size])
        Returns:
        vlad: the pooled vector of size: 'batch_size' x 'output_dim'
        """
        reshaped_input = input.view(-1, self.feature_size)
        activation = torch.matmul(reshaped_input, self.cluster_weights)

        if self.add_batch_norm:
            activation = self.batch_norm(activation)
        else:
            activation += self.cluster_biases

        activation = self.softmax(activation)
        activation = activation.view(-1, self.max_samples, self.cluster_size)
        activation = activation.permute(0, 2, 1)

        reshaped_input = reshaped_input.view(-1, self.max_samples, self.feature_size)
        vlad = torch.matmul(activation, reshaped_input)

        vlad = vlad.permute(0, 2, 1)
        vlad = F.normalize(vlad, dim=1, p=2)
        vlad = vlad.contiguous().view((-1, self.cluster_size * self.feature_size))
        vlad = F.normalize(vlad, dim=1, p=2)
        vlad = torch.matmul(vlad, self.hidden1_weights)

        if self.gating:
            vlad = self.context_gating(vlad)

        return vlad


class SoftDBoW(PoolingBaseModel):
    """Creates a Soft Deep Bag-of-Features class.
    """

    def __init__(
        self,
        feature_size,
        max_samples,
        cluster_size,
        output_dim,
        gating=True,
        add_batch_norm=True,
        is_training=True,
    ):
        super(SoftDBoW, self).__init__(
            feature_size=feature_size,
            max_samples=max_samples,
            cluster_size=cluster_size,
            output_dim=output_dim,
            gating=gating,
            add_batch_norm=add_batch_norm,
            is_training=is_training,
        )

        self.cluster_weights = nn.Parameter(
            torch.nn.init.normal_(
                torch.empty(self.feature_size, self.cluster_size),
                mean=0,
                std=(1 / math.sqrt(self.feature_size)),
            )
        )

        if add_batch_norm:
            self.cluster_biases = None
            self.batch_norm = nn.BatchNorm1d(self.cluster_size)
        else:
            self.cluster_biases = nn.Parameter(
                torch.nn.init.normal_(
                    torch.empty(self.cluster_size),
                    mean=0,
                    std=(1 / math.sqrt(self.feature_size)),
                )
            )
            self.batch_norm = None

        self.hidden1_weights = nn.Parameter(
            torch.nn.init.normal_(
                torch.empty(self.cluster_size, self.output_dim),
                mean=0,
                std=(1 / math.sqrt(self.cluster_size)),
            )
        )

        self.softmax = nn.Softmax(dim=-1)

        if gating:
            self.context_gating = GatingContext(
                self.output_dim, add_batch_norm=self.add_batch_norm
            )

    def forward(self, input):
        """Forward pass of a Soft-DBoW block.
        Args:
        reshaped_input: If your input is in that form:
        'batch_size' x 'max_samples' x 'feature_size'
        It should be reshaped in the following form:
        'batch_size*max_samples' x 'feature_size'
        by performing:
        reshaped_input = tf.reshape(input, [-1, features_size])
        Returns:
        bof: the pooled vector of size: 'batch_size' x 'output_dim'
        """
        reshaped_input = input.view(-1, self.feature_size)
        activation = torch.matmul(reshaped_input, self.cluster_weights)

        if self.add_batch_norm:
            activation = self.batch_norm(activation)
        else:
            activation += self.cluster_biases

        activation = self.softmax(activation)
        activation = activation.view(-1, self.max_samples, self.cluster_size)

        bof = activation.sum(1)
        bof = F.normalize(bof, dim=1, p=2)
        bof = torch.matmul(bof, self.hidden1_weights)

        if self.gating:
            bof = self.context_gating(bof)
        return bof


class NetFV(PoolingBaseModel):
    """Creates a NetFV class.
    """

    def __init__(
        self,
        feature_size,
        max_samples,
        cluster_size,
        output_dim,
        gating=True,
        add_batch_norm=True,
        is_training=True,
    ):
        super(NetFV, self).__init__(
            feature_size=feature_size,
            max_samples=max_samples,
            cluster_size=cluster_size,
            output_dim=output_dim,
            gating=gating,
            add_batch_norm=add_batch_norm,
            is_training=is_training,
        )

        self.cluster_weights = nn.Parameter(
            torch.nn.init.normal_(
                torch.empty(self.feature_size, self.cluster_size),
                mean=0,
                std=(1 / math.sqrt(self.feature_size)),
            )
        )

        self.covar_weights = nn.Parameter(
            torch.nn.init.normal_(
                torch.empty(self.feature_size, self.cluster_size),
                mean=0,
                std=(1 / math.sqrt(self.feature_size)),
            )
        )

        self.cluster_weights2 = nn.Parameter(
            torch.nn.init.normal_(
                torch.empty(1, self.feature_size, self.cluster_size),
                mean=0,
                std=(1 / math.sqrt(self.feature_size)),
            )
        )

        self.hidden1_weights = nn.Parameter(
            torch.nn.init.normal_(
                torch.empty(2 * self.feature_size * self.cluster_size, self.output_dim),
                mean=0,
                std=(1 / math.sqrt(self.cluster_size)),
            )
        )

        if add_batch_norm:
            self.cluster_biases = None
            self.batch_norm = nn.BatchNorm1d(self.cluster_size)
        else:
            self.cluster_biases = nn.Parameter(
                torch.nn.init.normal_(
                    torch.empty(self.cluster_size),
                    mean=0,
                    std=(1 / math.sqrt(self.feature_size)),
                )
            )
            self.batch_norm = None

        self.softmax = nn.Softmax(dim=-1)

        if gating:
            self.context_gating = GatingContext(
                self.output_dim, add_batch_norm=self.add_batch_norm
            )

    def forward(self, input):
        """Forward pass of a NetFV block.
        Args:
        reshaped_input: If your input is in that form:
        'batch_size' x 'max_samples' x 'feature_size'
        It should be reshaped in the following form:
        'batch_size*max_samples' x 'feature_size'
        by performing:
        reshaped_input = tf.reshape(input, [-1, features_size])
        Returns:
        fv: the pooled vector of size: 'batch_size' x 'output_dim'
        """
        reshaped_input = input.view(-1, self.feature_size)
        # This square might be unnecessary
        self.covar_weights = self.covar_weights ** 2
        eps = torch.tensor([1e-6], requires_grad=False)
        self.covar_weights = self.covar_weights + eps
        activation = torch.matmul(reshaped_input, self.cluster_weights)

        if self.add_batch_norm:
            activation = self.batch_norm(activation)
        else:
            activation += self.cluster_biases

        activation = self.softmax(activation)
        activation = activation.view(-1, self.max_samples, self.cluster_size)

        a_sum = activation.sum(-2, keepdim=True)
        a = a_sum * self.cluster_weights2
        activation = activation.permute(0, 2, 1)

        reshaped_input = reshaped_input.view(-1, self.max_samples, self.feature_size)

        fv1 = torch.matmul(activation, reshaped_input)
        fv1 = fv1.permute(0, 2, 1)

        # computing second order FV
        a2 = a_sum * (self.cluster_weights2 ** 2)
        b2 = fv1 * self.cluster_weights2
        fv2 = torch.matmul(activation, (reshaped_input ** 2))

        fv2 = fv2.permute(0, 2, 1)
        fv2 = a2 + fv2 + (-2 * b2)
        fv2 = fv2 / (self.covar_weights ** 2)
        fv2 = fv2 - a_sum

        fv2 = fv2.view(-1, self.cluster_size * self.feature_size)
        fv2 = F.normalize(fv2, dim=1, p=2)
        fv2 = fv2.view(-1, self.cluster_size * self.feature_size)
        fv2 = F.normalize(fv2, dim=1, p=2)

        fv1 = fv1 - a
        fv1 = fv1 / self.covar_weights

        fv1 = F.normalize(fv1, dim=1, p=2)
        fv1 = fv1.view(-1, self.cluster_size * self.feature_size)
        fv1 = F.normalize(fv1, dim=1, p=2)

        fv = torch.cat((fv1, fv2), 1)
        fv = torch.matmul(fv, self.hidden1_weights)

        if self.gating:
            fv = self.context_gating(fv)

        return fv
