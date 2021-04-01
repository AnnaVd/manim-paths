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
        
        thing = D
        grid = thing.grid()
        path = thing.path()
        labels = thing.labels()

        self.wait(1)
        self.play(ShowCreation(grid))
        self.play(ShowCreation(path))
        self.play(ShowCreation(labels))
        self.play(ShowCreation(D.circle_labels([1,2], color = MYGREEN)))
        self.wait(1)