import torch
from torch import nn
from torch.utils.data import Dataset
import pennylane as qml


import torch
from torch.utils.data import Dataset
import numpy as np

class SequenceDataset(Dataset):
    def __init__(self, dataframe, target, features, sequence_length=5):
        self.dataframe = dataframe  # ✅ required for __getitem__
        self.features = features
        self.target = target
        self.sequence_length = sequence_length

        # Ensure consistent float dtype for features
        feature_data = []
        for _, row in dataframe.iterrows():
            row_features = []
            for col in features:
                val = row[col]
                if isinstance(val, list):
                    row_features.extend(val)  # Flatten sequence
                else:
                    row_features.append(val)
            feature_data.append(row_features)

        self.X = torch.tensor(feature_data, dtype=torch.float32)
        self.y = torch.tensor(dataframe[self.target].values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        row_features = []

        for col in self.features:
            val = row[col]
            if isinstance(val, list):
                row_features.extend(val)  # Already a list → flatten
            elif isinstance(val, str) and val.startswith("["):
                # Convert stringified list → list of floats
                val = [float(x.strip()) for x in val.strip("[]").split(",")]
                row_features.extend(val)
            else:
                row_features.append(float(val))  # Single scalar value

        return torch.tensor(row_features, dtype=torch.float32), self.y[idx]



#classical lstm
import torch
import torch.nn as nn

class ShallowRegressionLSTM(nn.Module):
    def __init__(self, num_sensors, hidden_units, num_layers=1):
        super().__init__()
        self.num_sensors = num_sensors  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=num_sensors,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers,
        )

        self.linear = nn.Linear(in_features=self.hidden_units, out_features=1)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).to(x.device)

        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.linear(hn[0]).flatten()  # shape: (batch,)
        return torch.sigmoid(out)  # ensure output is in (0, 1) range


# Quantum LSTM
class QLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        n_qubits=4,
        n_qlayers=1,
        n_vrotations=3,
        batch_first=True,
        return_sequences=False,
        return_state=False,
        backend="default.qubit",
    ):
        super(QLSTM, self).__init__()
        self.n_inputs = input_size
        self.hidden_size = hidden_size
        self.concat_size = self.n_inputs + self.hidden_size
        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        self.n_vrotations = n_vrotations
        self.backend = backend  # "default.qubit", "qiskit.basicaer", "qiskit.ibm"

        self.batch_first = batch_first
        self.return_sequences = return_sequences
        self.return_state = return_state

        self.wires_forget = [f"wire_forget_{i}" for i in range(self.n_qubits)]
        self.wires_input = [f"wire_input_{i}" for i in range(self.n_qubits)]
        self.wires_update = [f"wire_update_{i}" for i in range(self.n_qubits)]
        self.wires_output = [f"wire_output_{i}" for i in range(self.n_qubits)]

        self.dev_forget = qml.device(self.backend, wires=self.wires_forget)
        self.dev_input = qml.device(self.backend, wires=self.wires_input)
        self.dev_update = qml.device(self.backend, wires=self.wires_update)
        self.dev_output = qml.device(self.backend, wires=self.wires_output)

        # self.dev_forget = qml.device(self.backend, wires=self.n_qubits)
        # self.dev_input = qml.device(self.backend, wires=self.n_qubits)
        # self.dev_update = qml.device(self.backend, wires=self.n_qubits)
        # self.dev_output = qml.device(self.backend, wires=self.n_qubits)

        def ansatz(params, wires_type):
            # Entangling layer.
            # ✅ Safe entangling layer — avoids using same qubit as control and target
            def ansatz(params, wires_type):
    # Safe entangling layer
            # Safer entanglement to avoid CNOT with same wires
                def ansatz(params, wires_type):
        # ✅ Safe CNOT chain
                    for i in range(self.n_qubits - 1):
                        qml.CNOT(wires=[wires_type[i], wires_type[i + 1]])

            # Variational layer
            for i in range(self.n_qubits):
                qml.RX(params[0][i], wires=wires_type[i])
                qml.RY(params[1][i], wires=wires_type[i])
                qml.RZ(params[2][i], wires=wires_type[i])


            # Variational layer
            for i in range(self.n_qubits):
                qml.RX(params[0][i], wires=wires_type[i])
                qml.RY(params[1][i], wires=wires_type[i])
                qml.RZ(params[2][i], wires=wires_type[i])


        def VQC(features, weights, wires_type):
            # Preproccess input data to encode the initial state.
            qml.templates.AngleEmbedding(features, wires=wires_type)
            ry_params = [torch.arctan(feature) for feature in features][0]
            rz_params = [torch.arctan(feature**2) for feature in features][0]
            for i in range(self.n_qubits):
                qml.Hadamard(wires=wires_type[i])
                qml.RY(ry_params[i], wires=wires_type[i])
                qml.RZ(ry_params[i], wires=wires_type[i])

            # Variational block.
            qml.layer(ansatz, self.n_qlayers, weights, wires_type=wires_type)

        def _circuit_forget(inputs, weights):
            VQC(inputs, weights, self.wires_forget)
            return [qml.expval(qml.PauliZ(wires=i)) for i in self.wires_forget]

        self.qlayer_forget = qml.QNode(
            _circuit_forget, self.dev_forget, interface="torch"
        )

        def _circuit_input(inputs, weights):
            VQC(inputs, weights, self.wires_input)
            return [qml.expval(qml.PauliZ(wires=i)) for i in self.wires_input]

        self.qlayer_input = qml.QNode(_circuit_input, self.dev_input, interface="torch")

        def _circuit_update(inputs, weights):
            VQC(inputs, weights, self.wires_update)
            return [qml.expval(qml.PauliZ(wires=i)) for i in self.wires_update]

        self.qlayer_update = qml.QNode(
            _circuit_update, self.dev_update, interface="torch"
        )

        def _circuit_output(inputs, weights):
            VQC(inputs, weights, self.wires_output)
            return [qml.expval(qml.PauliZ(wires=i)) for i in self.wires_output]

        self.qlayer_output = qml.QNode(
            _circuit_output, self.dev_output, interface="torch"
        )

        weight_shapes = {"weights": (self.n_qlayers, self.n_vrotations, self.n_qubits)}
        print(
            f"weight_shapes = (n_qlayers, n_vrotations, n_qubits) = ({self.n_qlayers}, {self.n_vrotations}, {self.n_qubits})"
        )

        self.clayer_in = torch.nn.Linear(self.concat_size, self.n_qubits)
        self.VQC = {
            "forget": qml.qnn.TorchLayer(self.qlayer_forget, weight_shapes),
            "input": qml.qnn.TorchLayer(self.qlayer_input, weight_shapes),
            "update": qml.qnn.TorchLayer(self.qlayer_update, weight_shapes),
            "output": qml.qnn.TorchLayer(self.qlayer_output, weight_shapes),
        }
        self.clayer_out = torch.nn.Linear(self.n_qubits, self.hidden_size)
        # self.clayer_out = [torch.nn.Linear(n_qubits, self.hidden_size) for _ in range(4)]

    def forward(self, x, init_states=None):
        """
        x.shape is (batch_size, seq_length, feature_size)
        recurrent_activation -> sigmoid
        activation -> tanh
        """
        if self.batch_first is True:
            batch_size, seq_length, features_size = x.size()
        else:
            seq_length, batch_size, features_size = x.size()

        hidden_seq = []
        if init_states is None:
            h_t = torch.zeros(batch_size, self.hidden_size)  # hidden state (output)
            c_t = torch.zeros(batch_size, self.hidden_size)  # cell state
        else:
            # for now we ignore the fact that in PyTorch you can stack multiple RNNs
            # so we take only the first elements of the init_states tuple init_states[0][0], init_states[1][0]
            h_t, c_t = init_states
            h_t = h_t[0]
            c_t = c_t[0]

        for t in range(seq_length):
            # get features from the t-th element in seq, for all entries in the batch
            x_t = x[:, t, :]

            # Concatenate input and hidden state
            v_t = torch.cat((h_t, x_t), dim=1)

            # match qubit dimension
            y_t = self.clayer_in(v_t)

            f_t = torch.sigmoid(
                self.clayer_out(self.VQC["forget"](y_t))
            )  # forget block
            i_t = torch.sigmoid(self.clayer_out(self.VQC["input"](y_t)))  # input block
            g_t = torch.tanh(self.clayer_out(self.VQC["update"](y_t)))  # update block
            o_t = torch.sigmoid(
                self.clayer_out(self.VQC["output"](y_t))
            )  # output block

            c_t = (f_t * c_t) + (i_t * g_t)
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)

class QShallowRegressionLSTM(nn.Module):
    def __init__(self, num_sensors, hidden_units, n_qubits=0, n_qlayers=1):
        super().__init__()
        self.num_sensors = num_sensors  # number of features
        self.hidden_units = hidden_units
        self.num_layers = 1

        self.lstm = QLSTM(
            input_size=num_sensors,
            hidden_size=hidden_units,
            batch_first=True,
            n_qubits=n_qubits,
            n_qlayers=n_qlayers,
        )

        self.linear = nn.Linear(in_features=self.hidden_units, out_features=1)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(
            self.num_layers, batch_size, self.hidden_units
        ).requires_grad_()
        c0 = torch.zeros(
            self.num_layers, batch_size, self.hidden_units
        ).requires_grad_()

        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.linear(hn).flatten()

        # ✅ Return sigmoid output for binary classification
        return torch.sigmoid(out)

