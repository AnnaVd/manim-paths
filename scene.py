from manim import*
from paths import* 

P = Path([1,0,1,0,1,1,0,0])
D = Drawing(P)
Po = Polyomino([1,1,0,0,0], [0,0,1,0,1])
Do = Drawing(Po)

class Path(Scene):

    def construct(self):
        
        grid = D.grid()
        path = D.path()

        self.wait(1)
        self.play(ShowCreation(grid))
        self.play(ShowCreation(path))
        
        self.wait(1)