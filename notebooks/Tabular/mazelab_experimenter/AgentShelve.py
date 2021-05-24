import inspect

import mazelab_experimenter.agents as agents


class AgentShelve:
    """ Static utility class for retrieving and defining agents to interface with various algorithms that can act in the OpenAI Gym interface. """
    
    _IMPLEMENTED = {
        'TabularQLearner': agents.TabularQLearning,
        'TabularQLearnerN': agents.TabularQLearningN,
        'TabularQLambda': agents.TabularQLambda,
        'TabularQET': agents.TabularQET,
        'TabularDynaQ': agents.TabularDynaQ,
        'MonteCarloQLearner': agents.MonteCarloQLearner, 
        'RandomAgent': agents.RandomAgent,
        'HierQ': agents.HierQ,
        'HierQV2': agents.HierQV2,
        'HierQN': agents.HierQN,
        'HierQTD': agents.HierQTD,
        'HierQTS': agents.HierQTS,
        'HierQLambda': agents.HierQLambda
    }
    
    @staticmethod
    def print_interface():
        """ Access the documentation of the basic Agent interface. """
        return help(agents.Agent)
    
    @staticmethod
    def get_keyword_args(agent: str) -> inspect.Signature:
        """
        Retrieve the class constructor arguments for a specified agent.
        
        :param agent: str One of the agent names within AgentShelve._IMPLEMENTED
        :see: mazelab.generators
        """
        assert agent in AgentShelve._IMPLEMENTED, f"Incorrect Agent specified: {agent}"
       
        return inspect.signature(AgentShelve._IMPLEMENTED[agent])
    
    @staticmethod
    def get_types():
        """ Get all available implemented agents defined in _IMPLEMENTED. :see: mazelab_experimenter.agents for the backend code. """
        return list(AgentShelve._IMPLEMENTED.keys())
    
    @staticmethod
    def retrieve(agent: str, keyword_arguments: dict) -> agents.Agent:
        """ 
        Instantiate and return an implemented agent algorithm.
        
        All agents are designed to interface with OpenAI gym environments.
        
        :param agent: str Name of the agent to retrieve.
        :param keyword_arguments: dict Class constructor arguments for the specified agent.
        :see: AgentShelve.get_types for all available agent implementations.
        :see: AgentShelve.get_keyword_args for the required constructor arguments of any agent.
        """
        assert agent in AgentShelve._IMPLEMENTED, f"Incorrect Agent specified: {agent}"
        
        return AgentShelve._IMPLEMENTED[agent](**keyword_arguments)
