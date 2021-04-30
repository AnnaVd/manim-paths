from paths import*

class titleslide(Scene):
    def construct(self):
        title = Tex('Two Delta conjecture implications').shift(2 * UP).scale(1.5)
        author = Tex('Anna Vanden Wyngaerd').scale(.8).shift(.7 * UP)
        sasha = Tex('joint work with Alessandro Iraci').scale(.6)
        path = Path(path = [1,1,0,1,0,0,1,0], labels = [1,2,3,4]).draw().shift(2*DOWN).scale(.8)
        ApplyMethod(author.scale, 0.8)
        self.add(title)
        self.add(author)
        self.add(path)
        self.add(sasha)
        self.wait(2)
        self.play(
            ShrinkToCenter(path),
            FadeOut(title),
            FadeOut(author),
            FadeOut(sasha)
        )
        self.wait(1)

class slide1(Scene):
    def construct(self):
        title = slide_title(Tex("My field in one sentence"))
        nexttitle = slide_title(Tex("Symmetric functions"))
        s = Tex("Looking for"," combinatorial formulas"," for", " interesting", " symmetric functions").scale(.8)
        fb1 = SurroundingRectangle(s[1], buff = .1).set_stroke(PINK)
        fb2 = SurroundingRectangle(s[3], buff = .1).set_stroke(PINK)
        fb3 = SurroundingRectangle(s[4], buff = .1).set_stroke(PINK)
        self.play(FadeIn(title))
        self.wait(1)
        self.play(Write(s))
        self.wait(1)
        self.play(Create(fb1))
        self.wait(1)
        self.play(fb1.animate.become(fb3))
        self.wait(1)
        self.play(fb1.animate.become(fb2))
        self.wait(1)
        self.play(
            s[4].animate.become(nexttitle),
            FadeOut(title),
            FadeOut(s[:4]),
            FadeOut(fb1)
        )
        self.wait(1)

class slide2(Scene):
    def construct(self):
        title = slide_title(Tex("Symmetric functions"))
        what = Tex(
            "Power series of bounded degree with \\\\",
            "$\\quad\\rightarrow$ coefficients in a field $\\mathbb{K}$\\\\",
            "$\\quad\\rightarrow$ in infinately many variables $x_1, x_2, \\dots$\\\\",
            "$\\quad\\rightarrow$ invariant by permutation of those variables",
            tex_environment="flushleft").scale(.8).align_to(title, LEFT).shift(2*UP)
        
        eg = Tex("Example  ", "$e_2 = x_1 x_2 + x_2x_3 + x_1x_3 + x_3x_4 + \\cdots$").scale(.8).shift(.5 * DOWN)
        eg[0].set_color(LIGHTBLUE)

        space = Tex("The space ", "$\\Lambda_{\\mathbb K}$", " of symmetric functions is naturally graded by homogeneous degree: $\\Lambda_{\\mathbb K} = \\oplus_{n\\in \\mathbb N}$", "$\\Lambda_{\\mathbb K}^{(n)}$").scale(.8)
        space[1].set_color(PINK)
        space[3].set_color(PINK)
        space.to_edge(DOWN)

        dim = MathTex("\\dim(\\Lambda_{\\mathbb K}^{(n)}) = \\text{\\# partitions } \\lambda \\vdash n").scale(.8).align_to(title, LEFT).shift(2.8*UP)

        self.add(title)
        for w in what:
            self.play(Write(w))
            self.wait(1)
        self.play(Write(eg))
        self.wait(1)
        self.play(Write(space))
        self.wait(1)
        self.play(FadeOut(what), FadeOut(eg), FadeOut(space[:3]), space[3].animate.become(dim))
        self.wait(1)

class slide3(Scene):
    def construct(self):
        title = slide_title(Tex("Symmetric functions"))
        
        dim = Tex("$\\dim(\\Lambda_{\\mathbb K}^{(n)})$ = $\\#$", " partitions ", "$\\lambda \\vdash n$").scale(.8).align_to(title, LEFT).shift(2.8*UP)

        partition = VGroup(
            MathTex(r"(\lambda_1,\dots, \lambda_k) \quad \lambda_i\in \mathbb N_0\\ \lambda_1\geq \cdots \geq \lambda_k \quad \sum_i \lambda_i = n"),
            MathTex("(3)").shift(1.5*DOWN + 2*LEFT), 
            YT([3]).draw().shift(2.8*DOWN + 2*LEFT).scale(.5),
            MathTex("(2,1)").shift(1.5*DOWN),
            YT([2,1]).draw().shift(2.8*DOWN).scale(.5),
            MathTex("(1,1,1)").shift(1.5*DOWN + 2*RIGHT),
            YT([1,1,1]).draw().shift(2.8*DOWN + 2*RIGHT).scale(.5), 
        )
        partition.scale(.7).to_corner(UP + RIGHT)
        partbox = SurroundingRectangle(partition, buff = .1).set_color(LIGHTBLUE)

        basis = Tex("Some classical basis of $\\Lambda_{\\mathbb K}^{(n)}$: ").scale(.8).align_to(partbox, DOWN).align_to(title, LEFT).shift(.1*RIGHT)

        ne = Tex("Elementary").scale(.8).set_color(LIGHTBLUE).shift(.5*DOWN)
        nh = Tex("Homogeneous").scale(.8).set_color(LIGHTBLUE).shift(.5*DOWN)
        np = Tex("Power").scale(.8).set_color(LIGHTBLUE).shift(.5*DOWN)
        ns = Tex("Schur").scale(.8).set_color(LIGHTBLUE).shift(.5*DOWN)

        fe = MathTex(r"e_{(2,1)} = e_2\cdot e_1 = (x_{1} x_{2} + x_{1} x_{3} + x_{2} x_{3}+\cdots)(x_1+x_2+x_3+ \cdots )").scale(.8).shift(1.5*DOWN)
        fh = MathTex(r"h_{(2,1)}= h_2\cdot h_1 = (x_{1}^{2} + x_{1} x_{2} + x_{2}^{2} +\cdots)(x_1+x_2+\cdots)").scale(.8).shift(1.5*DOWN)
        fp = MathTex(r"p_{(2,1)} = p_2\cdot p_1 =(x_{1}^{2}+ x_{2}^2 + x_3^2+\cdots)(x_{1} +x_{2}+x_{3}+\cdots)").scale(.8).shift(1.5*DOWN)
        fs = MathTex(
            r"s_{(2,1)} =", 
            r"x_{1}^{2} x_{2}",
            r"+ x_{1} x_{2}^{2}", 
            r"+ x_{1}^{2} x_{3}", 
            r" + 2 x_{1} x_{2} x_{3}",
            r"+ x_{2}^{2} x_{3}",
            r" + x_{1} x_{3}^{2}", 
            r"+ x_{2} x_{3}^{2}+\cdots"
            ).scale(.8).shift(1.5*DOWN)

       
        young = []
        young += [YT([2,1], labels= l).draw().scale(.7).shift(3*DOWN) for l in [[1,1,2], [1,2,2], [1,1,3],[2,2,3], [1,3,3], [2,3,3]]]
        young .insert(3,VGroup(
            YT([2,1], labels= [1,2,3]).draw().scale(.7).shift(3*DOWN + LEFT),
            YT([2,1], labels= [1,3,2]).draw().scale(.7).shift(3*DOWN + RIGHT)
        )) 
        
        self.add(title,dim)
        self.wait(1)
        self.play(dim[1].animate.set_color(LIGHTBLUE))
        self.play(Create(partition), Create(partbox))
        self.wait(1)
        self.play(Write(basis))
        self.wait(1)
        self.play(Write(ne), Write(fe)) 
        self.wait(1)
        self.play(ne.animate.become(nh), fe.animate.become(fh))
        self.wait(1)
        self.play(ne.animate.become(np), fe.animate.become(fp))
        self.wait(1)
        self.play(ne.animate.become(ns), fe.animate.become(fs[0]))
        self.wait(1)
        M, T = fs[1], young[0]
        self.play(Write(M),Create(T))
        self.wait()
        for mon, tab in zip(fs[2:], young[1:]):
            self.play(
                Write(mon),
                T.animate.become(tab)
            )
            self.wait(1)                
        self.play(*[FadeOut(o) for o in [T,ne,fs,fe,basis,partition, dim, partbox]])
        self.wait(4)
        
class slide4(Scene):
    def construct(self):
        title = slide_title(Tex("Symmetric functions"))
        mac = Tex(r"For $\mathbb K = \mathbb Q(q,t)$, the ", r"Macdonald polynomials, ", r"$H_\lambda$, form an important basis").scale(.8)
        mac[1].set_color(LIGHTBLUE)
        self.add(title)
        self.play(Write(mac))
        self.wait(1)
        self.play(FadeOut(mac))
        self.wait(1)