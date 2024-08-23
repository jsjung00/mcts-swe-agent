import math
import random
from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from collections import deque 


class Reflection(BaseModel): #TOOD: replace the reflection 
    reflections: str = Field(
        description="The critique and reflections on the sufficiency, superfluency,"
        " and general quality of the response"
    )
    score: int = Field(
        description="Score from 0-10 on the quality of the candidate response.",
        gte=0,
        lte=10,
    )
    found_solution: bool = Field(
        description="Whether the response has fully solved the question or task."
    )

    def as_message(self):
        return HumanMessage(
            content=f"Reasoning: {self.reflections}\nScore: {self.score}"
        )

    @property
    def normalized_score(self) -> float:
        return self.score / 10.0

class Node:
    def __init__(
            self,
            messages: list[BaseMessage],
            reflection: Optional[Reflection],
            parent: Optional['Node'],
            git_commit_hash: str,
            trajectory: list[str],
            env_state: Optional[str],
            observation: Optional[str],
            history: list[str],
            env_vars: dict[str, str],
            #value: float = 0.0,
    ):
        self.messages = messages 
        self.parent = parent 
        self.children = []
        self.value = 0 # TODO value
        self.visits = 0
        self.reflection = reflection 
        self.git_commit_hash = git_commit_hash
        self.trajectory = trajectory.copy()
        self.env_state = env_state
        self.observation = observation
        self.history = history.copy()
        self.env_vars = env_vars
        self.depth = parent.depth + 1 if parent is not None else 1 
        self._is_solved = reflection.found_solution if reflection else False 
        if self._is_solved:
            self._mark_tree_as_solved()
        if self.reflection:
            self.backpropagate(reflection.normalized_score)
    
    def __repr__(self) -> str:
        return (
            f"<Node value={self.value}, visits={self.visits},"
            f" solution={self.messages} reflection={self.reflection}/>"
        )
    
    @property
    def is_solved(self):
        return self._is_solved
    
    @property
    def is_terminal(self):
        return not self.children 
    
    @property
    def best_child(self):
        if not self.children:
            return None 
        all_nodes = self._get_all_children()
        return max(all_nodes, key=lambda child: child.upper_confidence_bound())
    
    @property
    def best_child_score(self):
        if not self.children:
            return None 
        return max(self.children, key=lambda child: int(child.is_solved)*child.value)
    
    def height(self):
        if self.children:
            return 1 + max([child.height for child in self.children])
        
    def upper_confidence_bound(self, exploration_weight=1.0):
        if self.parent is None:
            raise ValueError("Cannot obtain UCT from root node")
        if self.visits == 0:
            return self.value 
        
        average_reward = self.value / self.visits 
        exploration_term = math.sqrt(math.log(self.parent.visits) / self.visits)
        return average_reward + exploration_weight*exploration_term
    
    def backpropagate(self, reward:float):
        node = self 
        while node:
            node.visits += 1
            node.value = (node.value * (node.visits - 1) + reward) / node.visits 
            node = node.parent 
        
    def get_messages(self, include_reflections: bool = True):
        if include_reflections:
            return self.messages + [self.reflection.as_message()]
        return self.messages 
    
    #TODO: get_trajectory()

    def _get_all_children(self):
        all_nodes = []
        nodes = deque()
        nodes.append(self)
        while nodes:
            node = nodes.popleft()
            all_nodes.extend(node.children)
            for n in node.children:
                nodes.append(n)
        return all_nodes
    
    def get_best_solution(self):
        '''Return the best solution from within the current sub-tree'''
        all_nodes = [self] + self._get_all_children()
        best_node = max(
            all_nodes,
            key=lambda node: int(node.is_terminal and node.is_solved) * node.value 
        )
        return best_node
    
    def _mark_tree_as_solved(self):
        parent = self.parent 
        while parent:
            parent._is_solved = True 
            parent = parent.parent 


        




class TreeState(TypedDict):
    # this is the full MCTS tree 
    root: Node 
    input: str 
import math 

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state 
        self.parent = parent 
        self.children = []
        self.visits = 0
        self.value = 0
    
    def uct_value(self, exploration_weight=1.0):
        if self.visits == 0:
            return float('int')

        return (self.value / self.visits) + exploration_weight*math.sqrt(math.log(self.parent.visits)/self.visits)

    def best_child(self):
        if not self.children:
            return None 
        return max(self.children, key=lambda child: child.uct())

    def best_child_value(self):
        if not self.children:
            return None 
        return max(self.children, key=lambda child: child.value)
    
    def update(self, reward:float):
        self.visits += 1
        self.value += reward 

    

