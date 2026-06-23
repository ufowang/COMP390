
from .rnn_agent import RNNAgent
from .ff_agent import FFAgent
from .dueling_agent import DuelingAgent
from .central_rnn_agent import CentralRNNAgent

REGISTRY = {}

REGISTRY["rnn"] = RNNAgent
REGISTRY["ff"] = FFAgent
REGISTRY["dueling"] = DuelingAgent

REGISTRY["central_rnn"] = CentralRNNAgent
