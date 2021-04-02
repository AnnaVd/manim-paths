from manim import*
from paths import* 

P = Path([1,0,1,0,1,1,0,0], [2,3,1,4])
D = Drawing(P)
Po = Polyomino([1,1,0,0,0], 
[0,0,1,0,1],
array([[3,4],[1,None],[None,2]])
)
Do = Drawing(Po)


class Path(Scene):

    def construct(self):
        
        thing = Do
        grid = thing.grid()
        path = thing.path()
        labels = thing.labels()

        diag1 = thing.highlight_diagonal(1)
        diag2 = thing.highlight_diagonal(2)
        diag3 = thing.highlight_diagonal(3)
        diag4 = thing.highlight_diagonal(4)

        self.wait(1)
        self.play(ShowCreation(grid))
        self.play(ShowCreation(path))

        self.play(ShowCreation(diag1))
        self.play(ReplacementTransform(diag1, diag2))
        self.play(ReplacementTransform(diag2, diag3))
        self.play(ReplacementTransform(diag3, diag4))


        self.wait(1)