import implica

class GraphEngine :

    graph: implica.Graph

    def __init__(self) -> None:
        self.graph = implica.Graph(
            constants=[
                implica.Constant("K", "(A:*)->(B:*)->A"),
                implica.Constant("S", "((A:*) -> (B:*) -> (C:*)) -> (A -> B) -> A -> C")
            ]
        )

    def step(self, labels) -> None :
        pass
    
    def set_goal(self, label) -> None :
        pass 
    
    def clear_goal(self) -> None :
        pass

if __name__ == "__main__" :
    engine = GraphEngine()
    print(engine.graph.nodes())