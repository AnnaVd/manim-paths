from manim import*
from paths import* 

P = Path([1,0,1,0,1,1,0,0])
D = Drawing(P)

class Path(Scene):

    def construct(self):
        
        line = Line([0,0,0], [1,0,0])
        self.add(line)
        
        self.wait(2)