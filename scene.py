from manim import*
from paths import* 

P = Path([1,0,1,0,1,1,0,0], [2,3,1,4], rises = [1], valleys=[2])
Po = Polyomino([1,1,0,0,0], 
[0,0,1,0,1],
array([[3,4],[1,None],[None,2]])
)

class Path(Scene):

    def construct(self):

        self.wait(1)
        self.play(
            Create(P.draw_grid().to_edge(LEFT)),
            Create(Po.draw_grid().to_edge(RIGHT))
            )
        self.play(
            Create(P.draw_path().to_edge(LEFT)),
            Create(Po.draw_paths().to_edge(RIGHT))
            )
        self.play(
            Create(P.draw_labels().to_edge(LEFT)),
            Create(Po.draw_labels().to_edge(RIGHT))
        )
        self.play(
            Create(P.circle_labels([1,2]).to_edge(LEFT)),
            Create(Po.circle_labels([(1,2)]).to_edge(RIGHT))
        )
        self.play(
            Create(P.draw_decorations().to_edge(LEFT))
        )
        self.play(
            Create(P.highlight_diagonal(-1).to_edge(LEFT)),
            Create(Po.highlight_diagonal(3).to_edge(RIGHT))
        )
        
        #self.play(ShowSubmobjectsOneByOne(D22))
        #self.play(ShowCreation(grid))
        #self.play(ShowCreation(path))

        #self.play(ShowCreation(diag1))
        #self.play(ReplacementTransform(diag1, diag2))
        #self.play(ReplacementTransform(diag2, diag3))
        #self.play(ReplacementTransform(diag3, diag4))


        self.wait(1)