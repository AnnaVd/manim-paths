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

# one sentence
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

# symfun what
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

#symfun basis
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

#symfun macdonald        
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

#interesting symfun: Frobenius
class slide5(Scene):
    def construct(self): 
        oldtitle = slide_title(Tex("Symmetric functions"))
        title = slide_title(Tex(r"\textit{Interesting} symmetric functions"))
        newtitle = slide_title(Tex(r"Bi-graded representations of $\mathfrak{S}_n$"))

        frob = Tex(r"The ", r"Frobenius map ", r"is an isomorphism").scale(.8).align_to(title, LEFT + DOWN).shift(DOWN + .1*RIGHT)
        frob[1].set_color(PINK)


        frobmap = MathTex(r"\mathcal F : \text{class functions on } \mathfrak S_n \leftrightarrow \Lambda_{\mathbb C}^{(n)}").scale(.8).align_to(frob, DOWN).shift(DOWN)

        cor = Tex(r"which gives a correspondence between").scale(.8).align_to(frob, LEFT + DOWN).shift(2*DOWN)

        line= Line([0,-3,0], [0,0,0])

        irr = Tex(r"Irreducible representations of $\mathfrak S_n$").scale(.8).next_to(line,LEFT).shift(.5*LEFT)
        shu = Tex(r"Schur functions $\{s_\lambda\}_{\lambda \vdash n}$").scale(.8).next_to(line, RIGHT).shift(.5*RIGHT)

        rep = Tex(r"Any representation of $\mathfrak S_n$").scale(.8).next_to(line,LEFT).shift(.5*LEFT)
        shup = Tex(r"Positive integer combinations of \\ Schur functions \\ $\mathbb N\{s_\lambda\}_{\lambda \vdash n}$").scale(.8).next_to(line, RIGHT).shift(.5*RIGHT)

        birep = Tex(r"$V = \oplus_{i,j\in \mathbb N} V^{(i,j)}$\\", r"bi-graded representations of $\mathfrak S_n$").scale(.8).next_to(line,LEFT).shift(.5*LEFT)
        bishup = Tex(r"$q^i t^j\mathcal F (\chi^{V^{(i,j)} })$\\", r"Elements of $\mathbb N[q,t] \{s_\lambda\}_{\lambda \vdash n}$ ").scale(.8).next_to(line, RIGHT).shift(.5*RIGHT)

        pos = Tex("Schur positivity").set_color(PINK).to_edge(DOWN)

        self.play(ReplacementTransform(oldtitle, title))
        self.wait(1)
        self.play(Write(frob),Write(frobmap))
        self.wait(1)
        self.play(Write(cor))
        self.wait(1)
        self.play(Create(line))
        self.play(Write(irr))
        self.play(Write(shu))
        self.wait(1)
        self.play(ReplacementTransform(irr, rep), ReplacementTransform(shu, shup))
        self.wait(1)
        self.play(ReplacementTransform(rep, birep[0]), ReplacementTransform(shup, bishup[0]))
        self.wait(1)
        self.play(Write(birep[1]), Write(bishup[1]))
        self.wait(1)
        self.play(FadeInFrom(pos, DOWN))
        self.wait(1)
        self.play(
            *[FadeOut(mob) for mob in [frob, frobmap, pos, bishup, birep[0],line, title, cor]],
            birep[1].animate.become(newtitle)
            )
        self.wait(1)

#rep theory
class slide6(Scene):
    def construct(self):
        title = slide_title(Tex(r"Bi-graded representations of $\mathfrak{S}_n$"))
        macpos = Tex(r"Theorem ",r"(modified) MacDonald polynomials are the Frobenius image of the Garsia-Haiman modules and thus Schur positive").scale(.8)
        macpos[0].set_color(LIGHTBLUE)
        diag = Tex(r"A bi-product of this result: ", r"the module of ", r"Diagonal Coinvariants").scale(.8)
        diag[2].set_color(LIGHTBLUE)
        R = Tex(r"Let $R_n=\mathbb{C}[$",r"$x_1,\dots,x_n,$",r"$y_1,\dots,y_n$",r"$]$. ", r"For $\sigma\in \mathfrak S_n$ and $f\in R_n$").scale(.8).shift(1.5*UP)
        R[1].set_color(LIGHTBLUE)
        R[2].set_color(PINK)
        act = MathTex(r"\sigma\cdot f(x_1,\dots,x_n,y_1,\dots,y_n)= f(x_{\sigma(1)}, \dots, x_{\sigma(n)}, y_{\sigma(1)},\dots,y_{\sigma(n)})").scale(.8).shift(.5*UP)
        DC = MathTex(r"\mathcal{DC}_n := R_n/I^{\mathfrak{S}_n}").scale(.8).shift(.5*DOWN)
        
        nablathm = Tex(r"Theorem ",r"$\nabla e_n = \mathcal{F}_{q,t}(\mathcal{DC}_n)$").scale(.8).move_to(2*DOWN + 3*LEFT)
        nablathm[0].set_color(LIGHTBLUE)        
        nablathm.add(SurroundingRectangle(nablathm, buff = .2).set_color(LIGHTBLUE))
        nabla = Tex(r"$\nabla$ is a simple \\ Macdonald eigenoperator").scale(.8).move_to(2*DOWN + 3*RIGHT)

        sdiag = Tex(r"Super ",r"Diagonal Coinvariants").scale(.8).shift(2.5*UP)
        sdiag[0].set_color(PINK)
        sdiag[1].set_color(LIGHTBLUE)
        SR = Tex(r"Let $R_n=\mathbb{C}[$",r"$x_1,\dots,x_n,$",r"$y_1,\dots,y_n,$", r"$\theta_1,\dots,\theta_n$",r"$]$, ", r"with $\theta_i$ anti-commutative").scale(.8).shift(1.5*UP)
        SR[3].set_color(PINK)
        sact = Tex("Again, $\mathfrak S_n$ acts diagonally").scale(.8).shift(.5*UP)
        SDC = MathTex("{{ \\mathcal{SDC} }}_{n} := {{ R_n/I^{\\mathfrak{S}_n} }}").scale(.8).shift(.5*DOWN)
        SDCk = MathTex("{{ \\mathcal{SDC} }}_{n,k}  := ( {{ R_n/I^{\\mathfrak{S}_n} }} )^{(k)}").scale(.8).shift(.5*DOWN)
        hom = Tex(r"\\ Homogeneous $\theta$-degree $k$").scale(.8).shift(DOWN)

        deltathm = Tex(r"Conjecture ",r"$\Delta'_{e_{n-k-1} } e_n = \mathcal{F}_{q,t}(\mathcal{SDC}_{n-k})$").scale(.8).move_to(2.5*DOWN + 3.2*LEFT)
        deltathm[0].set_color(LIGHTBLUE)        
        deltathm.add(SurroundingRectangle(deltathm, buff = .2).set_color(LIGHTBLUE))
        delta = Tex(r"$\Delta_f, \Delta'_f$ for $f\in\Lambda$ are \\ Macdonald eigenoperators \\ generalising $\nabla$").scale(.8).move_to(2.5*DOWN + 3.5*RIGHT)

        self.add(title)
        self.wait(1)
        self.play(Write(macpos))
        self.wait(1)
        self.play(
            macpos.animate.shift(2*UP),
            Write(diag[0])
        )
        self.wait(1)
        self.play(Write(diag[1:]))
        self.play(
            FadeOut(macpos),
            FadeOut(diag[:2])
        )
        self.play(
            diag[2].animate.move_to(2.5*UP)
        )
        self.wait(1)
        self.play(Write(R[:4]))
        self.wait(1)
        self.play(
            R[1].animate.set_color(WHITE),
            R[2].animate.set_color(WHITE),
            Write(R[4:])
            )
        self.wait(1)
        self.play(Write(act)) 
        self.wait(1)
        self.play(Write(DC))
        self.wait(1)
        self.play(Write(nablathm))
        self.wait(1)
        self.play(Write(nabla))
        self.wait(1)
        self.play(
            *[FadeOut(m) for m in [nabla, nablathm, act, DC]]
        )
        self.play(
            diag[2].animate.become(sdiag),
            R.animate.become(SR)
        )
        self.wait(1)
        self.play(
            SR[3].animate.set_color(WHITE),
            diag[2][0].animate.set_color(LIGHTBLUE),
            Write(sact)
        )
        self.wait(1)
        self.play(Write(SDC))
        self.wait(1)
        #self.play(SDC.animate.become(SDCk))
        self.play(TransformMatchingTex(SDC, SDCk))
        self.play(Write(hom))
        self.wait(1)
        self.play(Write(deltathm))
        self.wait(1)
        self.play(Write(delta))
        self.wait(1)
        self.play(*[FadeOut(m) for m in self.mobjects])
        self.wait(1)

# recap
class slide7(Scene):
    def construct(self): 
        int = Tex("Interesting symmetric functions ", "are Schur positive elements of $\\Lambda_{\\mathbb Q(q,t)}$", " as they might be the Frobenius image of natural bi-graded representations of $\\mathfrak S_n$").scale(.8)
        int[0].set_color(LIGHTBLUE)

        su = Tex("For such functions, it is sometimes possible to find nice ", "combinatorial formulas", "\\\\ e.g. Shuffle theorem, Delta conjectures" ).scale(.8)
        su[1].set_color(LIGHTBLUE)

        ob = Tex("The combinatorial objects relevant to our discussion are ", "lattice paths").scale(.8)
        ob[1].set_color(LIGHTBLUE)

        coming = Tex("Coming up after the", " break",":").scale(.8).to_corner(DL)
        comb = Tex("combinatorics of lattice paths").scale(.8).next_to(coming, buff = .2)
        form = Tex("a bunch of combinatorial formulas").scale(.8).next_to(coming, buff = .2)
        imp = Tex("two implications").scale(.8).next_to(coming, buff = .2)
        nex = comb

        rect = SurroundingRectangle(VGroup(coming, form), buff = .5).set_color(LIGHTBLUE)

        

        brk = Tex("Break").scale(3)
        brkrect = SurroundingRectangle(brk, buff = .5).set_color(LIGHTBLUE)

        self.play(Write(int))
        self.wait(1)
        self.play(int.animate.shift(3*UP))
        self.play(Write(su[:2]))
        self.wait(1)
        self.play(Write(su[2]))
        self.wait(1)
        self.play(su.animate.next_to(int, DOWN))
        self.play(Write(ob))
        self.wait(1)        
        self.play(Create(rect))
        self.play(Write(coming))
        self.wait(1)
        self.play(Write(nex))
        self.wait(1)
        self.play(nex.animate.become(form))
        self.wait(1)
        self.play(nex.animate.become(imp))
        self.wait(1)
        self.play(*[FadeOut(m) for m in [nex,su,int,ob,coming[0], coming[2], rect]])
        self.play(coming[1].animate.become(brk))
        self.play(Create(brkrect))
        self.wait(3)
        self.play(FadeOut(coming[1]),Uncreate(brkrect))
        self.wait(1)

#combinatorics of lattice paths
spath = Path([0,1,1,0,0,1,0,1,0,1,1,1,0,1,0,0], labels = [2,4,2,1,3,4,6,5])
pspath = Path([0,1,1,0,0,1,0,1,0,1,1,1,0,1,0,0], labels = [2,4,2,0,3,4,6,0])
dpspath = Path([0,1,1,0,0,1,0,1,0,1,1,1,0,1,0,0], labels = [2,4,2,0,3,4,6,0], valleys = [2,4])
dpath = Path([1,1,0,1,0,0,1,1,0,1,0,1,1,0,0,0])
pic = 3.5*LEFT +.3*DOWN
wrd = 3.5*RIGHT +.3*DOWN


class slide8(Scene):
    def construct(self):
        #title
        title= slide_title(Tex("Lattice path combinatorics"))
        self.play(Write(title))
        self.wait(1)

        #grid and squarepath
        grid = dpath.draw_grid().shift(pic).scale(sc)
        squarepic = spath.draw_path().shift(pic).scale(sc)
        squaretxt = Tex("A", " square path", " of size 8").scale(.8).shift(wrd)
        squaretxt[1].set_color(LIGHTBLUE)
        txt = squaretxt
        path = deepcopy(squarepic)
        self.play(Create(grid))
        self.play(Write(txt))
        self.play(Create(path), run_time = 3, rate_func = linear)
        self.wait(1)
        
        #dyckpath
        dyckpic = dpath.draw_path().shift(pic).scale(sc)
        diag = dpath.draw_diagonal(0).shift(pic).scale(sc)
        dycktxt = Tex("A", " Dyck path", " of size 8").scale(.8).shift(wrd)
        dycktxt[1].set_color(LIGHTBLUE)
        self.play(
            txt.animate.become(dycktxt),
            path.animate.become(dyckpic),
            Create(diag)
        )
        self.wait(1)

        #return to squarepath
        self.play(path.animate.become(squarepic), FadeOut(txt), Uncreate(diag))
        self.wait(1)

        #labels
        labpic = spath.draw_labels().shift(pic).scale(sc)
        labtxt = Tex("A ","labelled ", "square path").scale(.8).shift(wrd)
        labtxt[1].set_color(LIGHTBLUE)
        txt = labtxt
        self.play(Write(txt), Create(labpic))
        self.wait(1)

        #partial labels
        plabpic = pspath.draw_labels().shift(pic).scale(sc)
        plabtxt = Tex("A ", "partially ", "labelled square path \\\\ $0$'s allowed!").scale(.8).shift(wrd)
        plabtxt[1].set_color(LIGHTBLUE)
        self.play(
            txt.animate.become(plabtxt),
            TransformMatchingShapes(labpic, plabpic)
        )
        self.wait(1)
        
        #monomial
        montxt = Tex("Monomial ","of a path $x_0\\mapsto 1$ \\\\ $x^P = x_2^2x_4^2x_6$").scale(.8).shift(wrd)
        montxt[0].set_color(LIGHTBLUE)
        self.play(txt.animate.shift(2*UP))
        self.play(Write(montxt))
        txt.add(montxt)
        self.wait(1)

        #area
        areatxt = Tex("Area").scale(.8).shift(wrd + UP).set_color(LIGHTBLUE)
        areatxt.add(Tex("Here area = 6").scale(.8).shift(wrd))
        areapic = pspath.higlight_area().shift(pic).scale(sc)
        areapic.add(pspath.draw_diagonal(-1).shift(pic).scale(sc))
        self.play(
            txt.animate.become(areatxt),
            Create(areapic)
        )
        self.wait(1)
        self.play(Uncreate(areapic), FadeOut(areatxt))

#combinatorics bis: dinv
class slide9(Scene):

    def construct(self):
        title= slide_title(Tex("Lattice path combinatorics"))
        self.add(title)
        self.add(pspath.draw().shift(pic).scale(sc))

        #dinv
        dinvtxt = Tex("Dinv: \\\\", "primary dinv", " + secondary dinv \\\\", "+ bonus dinv \\\\", "Here dinv = %d" %pspath.dinv()).scale(.8).shift(wrd)
        dinvtxt[0].set_color(LIGHTBLUE)
        lab0 = pspath.circle_labels([0]).shift(pic).scale(sc) 
        lab1 = pspath.circle_labels([1]).shift(pic).scale(sc) 
        lab2 = pspath.circle_labels([2]).shift(pic).scale(sc)
        lab3 = pspath.circle_labels([3]).shift(pic).scale(sc)
        lab4 = pspath.circle_labels([4]).shift(pic).scale(sc)
        #primary
        mone_diag = pspath.highlight_diagonal(-1).set_color(LIGHTBLUE).shift(pic).scale(sc)
  
        self.play(Write(dinvtxt[0]))
        self.wait(1)
        self.play(Create(mone_diag),Write(dinvtxt[1]))
        lab = deepcopy(lab0)
        self.play(
            Create(lab4),
            Create(lab)
        )
        self.wait(1)
        self.play(lab.animate.become(deepcopy(lab2)))
        self.wait(1)
        self.play(lab.animate.become(deepcopy(lab3)))
        self.wait(1)
        self.play(Uncreate(lab4), Uncreate(lab))
        self.wait(1)
        #secondary
        zero_diag = pspath.highlight_diagonal(0).set_color(GREEN).shift(pic).scale(sc)
        self.play(Create(zero_diag), Write(dinvtxt[2]))
        lab = deepcopy(lab2)
        self.play(
            Create(lab1),
            Create(lab)
        )
        self.wait(1)
        self.play(lab.animate.become(deepcopy(lab3)))
        self.wait(1)
        lab4 = pspath.circle_labels([4]).shift(pic).scale(sc)
        self.play(lab.animate.become(deepcopy(lab4)))
        self.wait(1)
        self.play(Uncreate(lab1), Uncreate(lab), FadeOut(mone_diag), FadeOut(zero_diag))
        self.wait(1)
        #bonus
        diag = pspath.draw_diagonal(0).shift(pic).scale(sc)
        self.play(Write(dinvtxt[3]))
        self.play(Create(diag))
        lab = lab0
        self.play(Create(lab))
        self.play(lab.animate.become(deepcopy(lab2)))
        self.wait(1)
        self.play(lab.animate.become(deepcopy(lab4)))
        self.wait(1)
        self.play(FadeOut(diag), FadeOut(lab))
        self.play(Write(dinvtxt[4]))
        self.wait(1)
        self.play(FadeOut(dinvtxt))
        self.wait(1)

#combinatorics bis: decorations
class slide10(Scene):

    def construct(self):
        title= slide_title(Tex("Lattice path combinatorics"))
        self.add(title)
        picture = pspath.draw().shift(pic).scale(sc)
        self.add(picture)

        valleys = pspath.higlight_steps([1,5,9]).shift(pic).scale(sc)
        valtxt = Tex("Choose some \\\\", "contractible valleys ", "to decorate").scale(.7).shift(wrd)
        valtxt[1].set_color(LIGHTBLUE)
        decs = dpspath.draw_decorations().shift(pic).scale(sc)
        stats = VGroup(
            Tex("Area remains unchanged"),
            Tex("ignore dinv pairs with left decorated label"),
            Tex(f"here, dinv = {dpspath.dinv()}")
        ).scale(.7).arrange(DOWN).shift(wrd + DOWN)

        self.play(Write(valtxt[0]))
        self.wait(.5)
        self.play(Create(valleys), Write(valtxt[1]))
        self.wait(1)
        self.play(Create(decs), Write(valtxt[2]))
        self.play(FadeOut(valleys))
        self.wait(1)
        self.play(valtxt.animate.shift(UP))
        self.play(Write(stats[0]))
        self.wait(1)
        self.play(Write(stats[1]))
        self.wait(1)
        self.play(Write(stats[2]))
        self.wait(1)

        picture.add(decs)
        self.play(*[FadeOut(m) for m in [title, valtxt, stats]], ShrinkToCenter(picture))
        self.wait(1)

#shuffle
class slide11(Scene):
    def construct(self): 
        title = slide_title(Tex("Combinatorial formulas"))

        form = Tex(r"Of the form \[\sum_{P \in \text{ set of LP} } q^{\text{dinv}(P)}t^{\text{area}(P)}x^P\]").scale(.8).to_corner(UP + RIGHT)
        box = SurroundingRectangle(form, buff = .2).set_color(LIGHTBLUE)

        shu = Tex("Shuffle Theorem: ", " formula for ", r"$\nabla e_n$ \\","using labelled Dyck paths of size $n$" ).scale(.8).align_to(title, LEFT + DOWN).shift(1.5*DOWN + .2*RIGHT)
        shu[0].set_color(LIGHTBLUE)

        labcomps = [[0,3,0], [0,2,1],[0,1,1,1]]
        auxS = [p for c in labcomps for p in paths(3, dyck = True, labels = c)]
        S = auxS[:15]
        ex = ["\\nabla e_3 = & ", S[0].monomial_string(), "+" + S[1].monomial_string()] 
        nl = 5
        for i in range(2,len(S)):
            if i % nl == 0:
                ex += ["+" + S[i].monomial_string() + "\\\\"]
            if i % nl == 1:
                ex += ["&+" + S[i].monomial_string()]
            if i %nl >1:
                ex += ["+" + S[i].monomial_string()]
        ex += ["+\\cdots"]
        mex = MathTex(*ex).scale(.8).shift(.5*UP)

        P = [VGroup(p.draw(),p.draw_diagonal(0)).to_edge(DOWN, buff = .3) for p in S]
        D = [Tex("dinv = %d" %p.dinv()).scale(.8).next_to(P[0], RIGHT).shift(.5*RIGHT + .5*UP) for p in S]
        A = [Tex("area = %d" %p.area()).scale(.8).next_to(P[0], RIGHT).shift(.5*RIGHT + .5*DOWN) for p in S]
       
        self.play(Write(title))
        self.play(Write(form), Create(box))
        self.wait(1)
        self.play(Write(shu))
        self.wait(1)
        self.play(Write(mex[0]))
        self.play(Write(mex[1]), Create(P[0]), Write(D[0]), Write(A[0]))
        self.wait(1)
        for i in range(1,len(S)):
            self.play(
                Write(mex[i+1]),
                ReplacementTransform(P[i-1], P[i]),
                ReplacementTransform(D[i-1], D[i]),
                ReplacementTransform(A[i-1], A[i])
            )
            self.wait(.3)
        self.play(Write(mex[-1]))
        self.wait(1)
        al =VGroup(mex, P[-1], D[-1], A[-1], shu, form, box)
        self.play(FadeOut(al))
        self.wait(1)

#delta
class slide12(Scene):
    def construct(self): 
        title = slide_title(Tex("Combinatorial formulas"))

        shu = Tex("Delta conjecture: ", " formula for ", r"$\Delta'_{e_{n-k-1} }e_n$ \\","using labelled Dyck paths of size $n$ \\\\ with $k$ decorated valleys" ).scale(.8).align_to(title, LEFT + DOWN).shift(1.7*DOWN + .3*RIGHT)
        shu[0].set_color(LIGHTBLUE)

        th = Tex(r"New Theta operators:\\ $\Delta'_{e_{n-k-1} }e_n= \Theta_k \nabla e_{n-k}$ \\ $\Theta_k$ $\leftrightarrow$ adding $k$ decorated steps").scale(.8).to_corner(RIGHT + UP)
        thbox = SurroundingRectangle(th, buff = .2).set_color(LIGHTBLUE)

        labcomps = [[0,3,0], [0,2,1],[0,1,1,1]]
        auxS = [p for c in labcomps for p in paths(3, dyck = True, labels = c, dvalleys = 1)]
        S = auxS[:15]

        print('FIRST PATH', S[0].aword, S[0].valleys)

        ex = ["\\Theta_1 \\nabla e_2 = & ", S[0].monomial_string(), "+" + S[1].monomial_string()] 
        nl = 5
        for i in range(2,len(S)):
            if i % nl == 0:
                ex += ["+" + S[i].monomial_string() + "\\\\"]
            if i % nl == 1:
                ex += ["&+" + S[i].monomial_string()]
            if i %nl >1:
                ex += ["+" + S[i].monomial_string()]
        ex += ["+\\cdots"]
        mex = MathTex(*ex).scale(.8).shift(.5*UP)

        P = [VGroup(p.draw(), p.draw_diagonal(0)).to_edge(DOWN, buff = .3) for p in S]
        D = [Tex("dinv = %d" %p.dinv()).scale(.8).next_to(P[0], RIGHT).shift(.5*RIGHT + .5*UP) for p in S]
        A = [Tex("area = %d" %p.area()).scale(.8).next_to(P[0], RIGHT).shift(.5*RIGHT + .5*DOWN) for p in S]
       
        self.add(title)
        self.wait(1)
        self.play(Write(shu))
        self.wait(1)
        self.play(Write(th), Create(thbox))
        self.wait(1)
        self.play(Write(mex[0]))
        self.play(Write(mex[1]), Create(P[0]), Write(D[0]), Write(A[0]))
        self.wait(1)
        for i in range(1,len(S)):
            self.play(
                Write(mex[i+1]),
                ReplacementTransform(P[i-1], P[i]),
                ReplacementTransform(D[i-1], D[i]),
                ReplacementTransform(A[i-1], A[i])
            )
            self.wait(.2)
        self.play(Write(mex[-1]))
        self.wait(1)
        al =VGroup(mex, P[-1], D[-1], A[-1], shu, th, thbox)
        self.play(FadeOut(al))

# gen Delta
class slide13(Scene):
    def construct(self): 
        title = slide_title(Tex("Combinatorial formulas"))

        shu = Tex("Generalised Delta conjecture:", " formula\\\\ for ", r"$\Delta_{h_m} \Theta_k\nabla e_{n-k}$ ","using partially labelled \\\\ Dyck paths of size $n+m$ with $k$ decorations \\\\ and $m$ zero labels" ).scale(.8).align_to(title, LEFT + DOWN).shift(2*DOWN + .3*RIGHT)
        shu[0].set_color(LIGHTBLUE)

        th = Tex(r" $\Delta_{h_m}$ \\ $\leftrightarrow$ \\ adding $m$ steps labelled 0").scale(.8).to_corner(UP + RIGHT)
        thbox = SurroundingRectangle(th, buff = .2).set_color(LIGHTBLUE)

        labcomps = [[1,2,0], [1,1,1],[1,0,2]]
        auxS = [p for c in labcomps for p in paths(3, dyck = True, dvalleys = 1, labels = c)]
        S = auxS[:15]
        ex = ["\\Delta_{h_1}\\Theta_1\\nabla e_1 = & ", S[0].monomial_string(), "+" + S[1].monomial_string()] 
        nl = 5
        for i in range(2,len(S)):
            if i % nl == 0:
                ex += ["+" + S[i].monomial_string()]
            if i % nl == 1:
                ex += ["&+" + S[i].monomial_string()]
            if i %nl >1:
                ex += ["+" + S[i].monomial_string()]
        ex += ["+\\cdots"]
        mex = MathTex(*ex).scale(.8)

        P = [VGroup(p.draw(), p.draw_diagonal(0)).to_edge(DOWN, buff = .3) for p in S]
        D = [Tex("dinv = %d" %p.dinv()).scale(.8).next_to(P[0], RIGHT).shift(.5*RIGHT + .5*UP) for p in S]
        A = [Tex("area = %d" %p.area()).scale(.8).next_to(P[0], RIGHT).shift(.5*RIGHT + .5*DOWN) for p in S]
       
        self.add(title)
        self.wait(1)
        self.play(Write(shu))
        self.wait(1)
        self.play(Write(th), Create(thbox))
        self.wait(1)
        self.play(Write(mex[0]))
        self.play(Write(mex[1]), Create(P[0]), Write(D[0]), Write(A[0]))
        self.wait(1)
        for i in range(1,len(S)):
            self.play(
                Write(mex[i+1]),
                ReplacementTransform(P[i-1], P[i]),
                ReplacementTransform(D[i-1], D[i]),
                ReplacementTransform(A[i-1], A[i])
            )
            self.wait(.2)
        self.play(Write(mex[-1]))
        self.wait(1)
        al =VGroup(mex, P[-1], D[-1], A[-1], shu, th, thbox)
        self.play(FadeOut(al))

#square
class slide14(Scene):
    def construct(self): 
        title = slide_title(Tex("Combinatorial formulas"))

        shu = Tex("Square Theorem: ", " formula for ", r"$(-1)^{n-1}\nabla p_n$"," using labelled square paths\\\\ of size $n$" ).scale(.8).align_to(title, LEFT + DOWN).shift(1.5*DOWN + .3*RIGHT)
        shu[0].set_color(LIGHTBLUE)

        labcomps = [[0,3,0], [0,2,1],[0,1,1,1]]
        auxS = [p for c in labcomps for p in paths(3, labels = c)]
        S = auxS[:15]
        ex = ["\\nabla p_3 = & ", S[0].monomial_string(), "+" + S[1].monomial_string()] 
        nl = 5
        for i in range(2,len(S)):
            if i % nl == 0:
                ex += ["+" + S[i].monomial_string() + "\\\\"]
            if i % nl == 1:
                ex += ["&+" + S[i].monomial_string()]
            if i %nl >1:
                ex += ["+" + S[i].monomial_string()]
        ex += ["+\\cdots"]
        mex = MathTex(*ex).scale(.8).shift(.5*UP)

        P = [p.draw().to_edge(DOWN, buff = .3) for p in S]
        D = [Tex("dinv = %d" %p.dinv()).scale(.8).next_to(P[0], RIGHT).shift(.5*RIGHT + .5*UP) for p in S]
        A = [Tex("area = %d" %p.area()).scale(.8).next_to(P[0], RIGHT).shift(.5*RIGHT + .5*DOWN) for p in S]
       
        self.add(title)
        self.wait(1)
        self.play(Write(shu))
        self.wait(1)
        self.play(Write(mex[0]))
        self.play(Write(mex[1]), Create(P[0]), Write(D[0]), Write(A[0]))
        self.wait(1)
        for i in range(1,len(S)):
            self.play(
                Write(mex[i+1]),
                ReplacementTransform(P[i-1], P[i]),
                ReplacementTransform(D[i-1], D[i]),
                ReplacementTransform(A[i-1], A[i])
            )
            self.wait(.2)
        self.play(Write(mex[-1]))
        self.wait(1)
        al =VGroup(mex, P[-1], D[-1], A[-1], shu)
        self.play(FadeOut(al))

# gen delta square
class slide15(Scene):
    def construct(self): 
        title = slide_title(Tex("Combinatorial formulas"))

        shu = Tex("Generalised Delta square Conjecture: ", " formula for ", r"$(-1)^{n-k-1}\Delta_{h_m}\Theta_k\nabla p_{n-k}$"," using labelled square paths of size $n+m$, $k$ decorations, $m$ zero labels" ).scale(.8).align_to(title, LEFT + DOWN).shift(1.5*DOWN + .3*RIGHT)
        shu[0].set_color(LIGHTBLUE)

        labcomps = [[1,2,0], [1,1,1],[1,0,2]]
        auxS = [p for c in labcomps for p in paths(3, dvalleys = 1, labels = c)]
        S = auxS[:15]
        ex = ["-\\Delta_{h_1}\\Theta_1\\nabla p_1 = & ", S[0].monomial_string(), "+" + S[1].monomial_string()] 
        nl = 5
        for i in range(2,len(S)):
            if i % nl == 0:
                ex += ["+" + S[i].monomial_string() + "\\\\"]
            if i % nl == 1:
                ex += ["&+" + S[i].monomial_string()]
            if i %nl >1:
                ex += ["+" + S[i].monomial_string()]
        ex += ["+\\cdots"]
        mex = MathTex(*ex).scale(.8).shift(.5*UP)

        P = [p.draw().to_edge(DOWN, buff = .3) for p in S]
        D = [Tex("dinv = %d" %p.dinv()).scale(.8).next_to(P[0], RIGHT).shift(.5*RIGHT + .5*UP) for p in S]
        A = [Tex("area = %d" %p.area()).scale(.8).next_to(P[0], RIGHT).shift(.5*RIGHT + .5*DOWN) for p in S]
       
        self.add(title)
        self.wait(1)
        self.play(Write(shu))
        self.wait(1)
        self.play(Write(mex[0]))
        self.play(Write(mex[1]), Create(P[0]), Write(D[0]), Write(A[0]))
        self.wait(1)
        for i in range(1,len(S)):
            self.play(
                Write(mex[i+1]),
                ReplacementTransform(P[i-1], P[i]),
                ReplacementTransform(D[i-1], D[i]),
                ReplacementTransform(A[i-1], A[i])
            )
            self.wait(.2)
        self.play(Write(mex[-1]))
        self.wait(1)
        al =VGroup(mex, P[-1], D[-1], A[-1], shu)
        self.play(FadeOut(al))

# two implications
class slide16(Scene):
    def construct(self): 
        title = slide_title(Tex("Two implications"))
        newtitle = slide_title(Tex("Delta $\\Rightarrow$ generalised Delta"))
        start = Tex("Finally: the ", "two implications ", "promised in the title").scale(.8)

        one = Tex("1. ", "Delta conjecture $\\Rightarrow$ generalised Delta conjecture\\\\", "Adding 0 labels").scale(.8).shift(2*UP)
        one[0].set_color(LIGHTBLUE)
        
        two = Tex("2. ", "(generalised) Delta conjecture $\\Rightarrow$ (generalised) Delta square conjecture\\\\", "from Dyck to square paths").scale(.8)
        two[2].align_to(two[1])
        two[0].set_color(LIGHTBLUE)

        rem = Tex("The remainder of the talk is a sketch of the combinatorial side of the proofs").scale(.8).to_edge(DOWN)

        self.wait(1)
        self.play(Write(start))
        self.wait(1)
        self.play(FadeOut(VGroup(start[0], start[2])), ReplacementTransform(start[1], title))
        self.play(Write(one))
        self.wait(1)
        self.play(Write(two))
        self.wait(1)
        self.play(Write(rem))
        self.wait(1)
        self.play(FadeOut(VGroup(title,two,one[0], one[2], rem)), ReplacementTransform(one[1], newtitle))
        self.wait(1)

# delta -> gen delta
sc = 0.8
P0 = Path(path = [1,1,0,1,0,0,1,0,1,1,0,0,1,0,1,0,1,1,0,0], labels = [2,3,4,4,1,3,4,2,3,4], valleys = [2,6,8])
P1 = Path(path = [1,1,0,0,1,0,1,0,1,1,0,0,1,0,1,0,1,0,1,0], labels = [2,3,0,4,1,3,4,2,3,0], valleys = [2,6,8])
P2 = Path(path = [1,1,0,0,1,0,1,1,0,0,1,0,1,0,1,0], labels = [2,3,0,1,3,2,3,0], valleys = [2,6])
        
class slide17(Scene):
    
    def construct(self): 
        title = slide_title(Tex("Delta $\\Rightarrow$ generalised Delta"))
        
        p0, lp0, dp0 = P0.draw_path().shift(pic).scale(sc), P0.draw_labels().shift(pic).scale(sc), P0.draw_decorations().shift(pic).scale(sc)
        p1, lp1, dp1 = P1.draw_path().shift(pic).scale(sc), P1.draw_labels().shift(pic).scale(sc), P1.draw_decorations().shift(pic).scale(sc)
        p2, lp2, dp2 = P2.draw_path().shift(pic).scale(sc), P2.draw_labels().shift(pic).scale(sc), P2.draw_decorations().shift(pic).scale(sc)
        
        biggrid = P0.draw_grid().shift(pic).scale(sc)
        smallgrid = P2.draw_grid().shift(pic).scale(sc)

        bigdiag = P0.draw_diagonal().shift(pic).scale(sc)
        smalldiag = P2.draw_diagonal().shift(pic).scale(sc)

        maxup = P0.circle_labels([2,9]).shift(pic).scale(sc)
        maxdiag = P0.circle_labels([3,6]).shift(pic).scale(sc)
        

        perp = Tex("Select maximal labels (apply $h_j^\\perp$)").scale(.8).shift(wrd + 2.5 * UP)
        if1 = Tex("- Strictly above the line $x=y$").scale(.8).shift(wrd + 2*UP)
        if2 = Tex("- On the line $x=y$").scale(.8).shift(wrd + 2*UP)
        stat1 = Tex("area changes predictably\\\\ dinv constant").scale(.8).shift(wrd + 2* DOWN)
        stat2 = Tex("dinv changes predictably\\\\ area constant").scale(.8).shift(wrd + 2* DOWN)

        step = .8
        bpush = VGroup(
            Line([0,0,0], [0,step,0]).set_stroke(color = PINK, width = 6, opacity = 1),
            Line([0,step,0], [step, step, 1]).set_stroke(color = PINK, width = 6, opacity = 1), 
            Tex("M").scale(step).shift([.5*step, .5*step,0])
        ).shift(1.75*RIGHT)
        apush = VGroup(
            Line([0,0,0], [step,0,0]).set_stroke(color = PINK, width = 6, opacity = 1),
            Line([step,0,0], [step, step, 1]).set_stroke(color = PINK, width = 6, opacity = 1), 
            Tex("0").scale(step).shift([1.5*step, .5*step,0])
        ).shift(1.75*RIGHT)
        bdpush = VGroup(
            Line([0,0,0], [0,step,0]).set_stroke(color = PINK, width = 6, opacity = 1),
            Line([0,step,0], [step, step, 1]).set_stroke(color = PINK, width = 6, opacity = 1), 
            Tex("M").scale(step).shift([.5*step, .5*step,0]),
            Tex("$\\bullet$").scale(step).shift([-.5*step, .5*step,0])
        ).shift(5.25*RIGHT)
        adpush = VGroup(
            Line([0,0,0], [step,0,0]).set_stroke(color = PINK, width = 6, opacity = 1),
            Line([step,0,0], [step, step, 1]).set_stroke(color = PINK, width = 6, opacity = 1), 
            Tex("0").scale(step).shift([1.5*step, .5*step,0]),
            Tex("$\\bullet$").scale(step).shift([.5*step, .5*step,0])
        ).shift(5.25*RIGHT)

        dia = VGroup(
            Line([0,0,0], [0,step,0]).set_stroke(color = PINK, width = 6, opacity = 1),
            Line([0,step,0], [step, step, 1]).set_stroke(color = PINK, width = 6, opacity = 1),
            Line([0,0,0], [step, step, 0]).set_stroke(opacity = .5),
            Tex("M").scale(step).shift([.5*step, .5*step,0])
        ).shift(1.75*RIGHT)
        ddia = VGroup(
            Line([0,0,0], [0,step,0]).set_stroke(color = PINK, width = 6, opacity = 1),
            Line([0,step,0], [step, step, 1]).set_stroke(color = PINK, width = 6, opacity = 1),
            Line([0,0,0], [step, step, 0]).set_stroke(opacity = .5),
            Tex("M").scale(step).shift([.5*step, .5*step,0]),
            Tex("$\\bullet$").scale(step).shift([-.5*step, .5*step,0])
        ).shift(5.25*RIGHT)
        dot = Dot().shift(1.75*RIGHT)
        ddot = Dot().shift(5.25*RIGHT)

        self.add(title)
        self.wait(1)
        self.play(Create(VGroup(biggrid, bigdiag, p0, lp0, dp0)))
        self.wait(1)
        self.play(Write(perp), Create(maxup), Create(maxdiag))
        self.wait(1)
        self.play(Write(if1), FadeOut(perp), maxup.animate.set_color(GREEN))
        self.wait(1)
        self.play(Create(bpush),Create(bdpush))
        self.wait(1)
        self.play(bpush.animate.become(apush), bdpush.animate.become(adpush))
        self.wait(1)
        self.play(
            p0.animate.become(p1), 
            lp0.animate.become(lp1), 
            dp0.animate.become(dp1),
            FadeOut(maxup), 
            run_time = 3)
        self.wait(1)
        self.play(Write(stat1))
        self.play(*[FadeOut(m) for m in [bpush, bdpush, if1,stat1]])
        self.play(Write(if2), maxdiag.animate.set_color(GREEN))
        self.wait(1)
        self.play(Create(dia), Create(ddia))
        self.wait(1)
        self.play(dia.animate.become(dot), ddia.animate.become(ddot))
        self.wait(1)
        self.play(
            p0.animate.become(p2), 
            lp0.animate.become(lp2), 
            dp0.animate.become(dp2),
            FadeOut(maxdiag), 
            run_time = 3)
        self.wait(1)
        self.play(biggrid.animate.become(smallgrid), bigdiag.animate.become(smalldiag))
        self.wait(1)
        self.play(Write(stat2))
        self.play(
            *[FadeOut(m) for m in [dia, ddia, if2,stat2]]
            )
        self.wait(1)
        self.play(
            FadeOut(title),
            *[ShrinkToCenter(m) for m in [p0,lp0,dp0,biggrid,bigdiag]]
        )
        self.wait(1)

# delta -> delta square
class slide18(Scene):
    def construct(self): 
        title = slide_title(Tex("Delta $\\Rightarrow$ Delta square"))
        strat1 = Tex("Build from scratch the set $S$ \\\\ of square paths with a fixed set of \\\\ (decorated) labels in each diagonal").scale(.8).shift(2.5*UP + 3.5*RIGHT)
        strat2 = Tex("get factorisation of \[ \\sum_{P\\in S}q^{\\text{dinv}(P)}t^{\\text{area}(P)}x^P \]").scale(.8).next_to(strat1, DOWN)
        strat3 = Tex("makes it clear how to shift \\\\ all labels to one diagonal higher").scale(.8).next_to(strat2, DOWN)
        strat4 = Tex("$\\rightarrow$ square paths in terms of Dyck paths").scale(.8).next_to(strat3, DOWN)


        
        p = Path(path = [0,1,0,1,0,1,1,0,1,1,0,0,1,1,0,0], labels = [1,2,1,3,2,4,2,4], valleys = [1])
        sp = Path(path = [1,0,1,0,1,1,0,1,1,0,0,1,1,0,0,0], labels = [1,2,1,3,2,4,2,4], valleys = [1])

        grid = p.draw_grid().shift(3.5*LEFT + .5*UP).scale(.8)
        path = p.draw_path().shift(3.5*LEFT + .5*UP).scale(.8)
        labels = p.draw_labels().shift(3.5*LEFT + .5*UP).scale(.8)
        decs = p.draw_decorations().shift(3.5*LEFT + .5*UP).scale(.8)

        m1 = p.highlight_diagonal(-1).set_color(GREEN).shift(3.5*LEFT + .5*UP).scale(.8)
        m2 = p.highlight_diagonal(0).set_color(GREEN).shift(3.5*LEFT+ .5*UP).scale(.8)
        m3 = p.highlight_diagonal(1).set_color(GREEN).shift(3.5*LEFT+ .5*UP).scale(.8) 
        sm = sp.draw().shift(3.5*LEFT+ .5*UP).scale(.8)

        dw = MathTex("&1\\quad 1 \\quad \\bullet 2& \\\\", "&2\\quad 2 \\quad 3 \\\\", "&4\\quad 4"
            ).scale(.6).next_to(m1, DOWN)

        self.play(Write(title))
        self.play(Write(strat1), Create(VGroup(grid, path, labels, decs)))
        self.wait(1)
        self.play(FadeIn(m1), Write(dw[0]))
        self.wait(1)
        self.play(FadeOut(m1), FadeIn(m2), Write(dw[1]))
        self.wait(1)
        self.play(FadeOut(m2), FadeIn(m3), Write(dw[2]))
        self.wait(1)
        self.play(FadeOut(m3))
        self.play(Write(strat2))
        self.wait(1)
        self.play(Write(strat3))
        self.wait(1)
        self.play(
            path.animate.become(sp.draw_path().shift(3.5*LEFT + .5*UP).scale(.8)),
            labels.animate.become(sp.draw_labels().shift(3.5*LEFT + .5*UP).scale(.8)),
            decs.animate.become(sp.draw_decorations().shift(3.5*LEFT + .5*UP).scale(.8)),
            run_time = 3
        )
        self.wait(1)
        self.play(Write(strat4))
        self.wait(1)
        self.play(FadeOut(VGroup(grid, path, decs, labels, strat1,strat2,strat3,strat4, dw)))

#tree
class slide19(Scene):
    def construct(self): 
        title = slide_title(Tex("Delta $\\Rightarrow$ Delta square"))
        #sub = Tex("Constructing all paths with diagonal word: $11(\\bullet 2)\\quad 223 \\quad 4$").align_to(title,LEFT).scale(.8)

        p0 = MathTex("\\emptyset")

        t1 = Tex("$3$ in $0$-diagonal").scale(.8)
        p1 = Path(path = [1,0], labels = [3]).draw().scale(.55)

        t2 = Tex("$2,2$ in $0$-diagonal").scale(.8)
        p21 = Path(path = [1,0,1,0,1,0], labels = [3,2,2]).draw().scale(.55) #chosen
        p22 = Path(path = [1,0,1,0,1,0], labels = [2,3,2]).draw().scale(.55)
        p23 = Path(path = [1,0,1,0,1,0], labels = [2,2,3]).draw().scale(.55)

        t3 = Tex("$4,4$ in $1$-diagonal").scale(.8)
        p31 = Path(path = [1,0,1,0,1,1,0,1,0,0], labels = [3,2,2,4,4]).draw().scale(.55)
        p32 = Path(path = [1,0,1,1,0,0,1,1,0,0], labels = [3,2,4,2,4]).draw().scale(.55) #chosen
        p33 = Path(path = [1,0,1,1,0,1,0,0,1,0], labels = [3,2,4,4,2]).draw().scale(.55)
        p34 = Path(path = [1,1,0,0,1,0,1,1,0,0], labels = [3,4,2,2,4]).draw().scale(.55)

        t4 = Tex("$1,1$ in $(-1)$-diagonal").scale(.8)
        p41 = Path(path = [0,1,0,1,1,0,1,1,0,0,1,1,0,0], labels = [1,1,3,2,4,2,4]).draw().scale(.55) #chosen
        p42 = Path(path = [0,1,1,0,0,1,1,1,0,0,1,1,0,0], labels = [1,3,1,2,4,2,4]).draw().scale(.55) 
        p43 = Path(path = [1,0,0,1,0,1,1,1,0,0,1,1,0,0], labels = [3,1,1,2,4,2,4]).draw().scale(.55)
        
        t5 = Tex("$\\bullet 2$ in $(-1)$-diagonal").scale(.8)
        p51 = Path(path = [0,1,0,1,0,1,1,0,1,1,0,0,1,1,0,0], labels = [2,1,1,3,2,2,4,4], valleys = [0]).draw().scale(.55)
        p52 = Path(path = [0,1,0,1,0,1,1,0,1,1,0,0,1,1,0,0], labels = [1,2,1,3,2,2,4,4], valleys = [1]).draw().scale(.55)
        p53 = Path(path = [0,1,0,1,0,1,1,0,1,1,0,0,1,1,0,0], labels = [1,1,2,3,2,2,4,4], valleys = [2]).draw().scale(.55)
        
        top = [0,2,0]
        
        
        d0 = Tex("dinv: + 0").scale(.8)
        d1 = Tex("dinv: + 1").scale(.8)
        d2 = Tex("dinv: + 2").scale(.8)
        d2bis = Tex("dinv: + 2").scale(.8)
        #d3 = Tex("dinv: + 3").scale(.8)
        dots = Tex("$\\cdots$")

        self.add(title)
        self.wait(1)
        
        bot = 0
        self.play(Write(p0.shift(top)))
        self.wait(1)
        self.play(Write(t1.next_to(p0,DOWN)))
        self.play(Create(p1.move_to([0,bot,0])), Write(d0.next_to(p1, DOWN)))
        self.wait(1)
        self.play(FadeOut(VGroup(p0,d0,t1)))
        self.play(ApplyMethod(p1.move_to, top))
        self.wait(1)

        bot = -1
        self.play(Write(t2.next_to(p1,DOWN)))
        self.wait(1)
        self.play(Create(p21.move_to([-3,bot,0])), Write(d0.next_to(p21, DOWN, buff = .1)))
        self.play(Create(p22.move_to([0,bot,0])), Write(d1.next_to(p22, DOWN, buff = .1)))
        self.play(Create(p23.move_to([3,bot,0])), Write(d2.next_to(p23, DOWN, buff = .1)))
        self.wait(1)
        self.play(FadeOut(VGroup(p22,p23,d0,d1,d2,t2, p1)))
        self.play(ApplyMethod(p21.move_to, top))
        self.wait(1)

        self.play(Write(t3.next_to(p21,DOWN)))
        self.wait(1)
        self.play(Create(p31.move_to([-4.5,bot,0])), Write(d0.next_to(p31, DOWN, buff = .1)))
        self.play(Create(p32.move_to([-1.5, bot, 0])), Write(d1.next_to(p32, DOWN, buff = .1)))
        self.play(Create(p33.move_to([1.5,bot,0])), Write(d2.next_to(p33, DOWN, buff = .1)))
        self.play(Create(p34.move_to([4.5,bot,0])), Write(d2bis.next_to(p34, DOWN, buff = .1)))
        self.play(Write(dots.move_to([6,bot,0])))
        self.wait(1)
        self.play(FadeOut(VGroup(p31,p33,p34,d0,d1,d2,d2bis,p21,t3,dots)))
        self.play(ApplyMethod(p32.move_to,top))
        self.wait(1)

        bot = -1.5
        self.play(Write(t4.next_to(p32,DOWN)))
        self.wait(1)
        self.play(Create(p41.move_to([-4,bot,0])), Write(d0.next_to(p41, DOWN, buff = .1)))
        self.play(Create(p42.move_to([0, bot, 0])), Write(d1.next_to(p42, DOWN, buff = .1)))
        self.play(Create(p43.move_to([4,bot,0])), Write(d2.next_to(p43, DOWN, buff = .1)))
        self.play(Write(dots.move_to([6,bot,0])))
        self.wait(1)
        self.play(FadeOut(VGroup(p42,p43,d0,d1,d2,p32,t4,dots)))
        top = [0,2.3,0]
        self.play(ApplyMethod(p41.move_to, top))
        self.wait(1)

        self.play(Write(t5.next_to(p41,RIGHT)))
        self.wait(1)
        self.play(Create(p51.move_to([-4.5,bot,0])), Write(d0.next_to(p51, DOWN, buff = .1)))
        self.play(Create(p52.move_to([0, bot, 0])), Write(d1.next_to(p52, DOWN, buff = .1)))
        self.play(Create(p53.move_to([4.5,bot,0])), Write(d2.next_to(p53, DOWN, buff = .1)))
        self.play(Write(dots.move_to([6.7,bot,0])))
        self.wait(1)
        self.play(FadeOut(VGroup(p51,p52,p53,d0,d1,d2,p41,t5,dots,title)))
        self.wait(1)    

# thanks
class slide19(Scene):
    def construct(self): 
        self.play(Write(Tex("Thank you for your attention!")))