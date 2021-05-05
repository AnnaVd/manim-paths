from itertools import combinations, product
from more_itertools.more import distinct_combinations
from multiset import Multiset as multiset
from numpy import array, diag, floor, gcd
from manim import*
from copy import deepcopy

from pygments import highlight

from permtools import *  # pylint: disable=unused-wildcard-import

# ANIMATION CONFIGURATION
# custom colors
#PINK = "#E20851"
config.background_color = "#003361"
PINK = "#E94E6E" 
GREEN = "#51e208"
LIGHTBLUE = "#64FBF9"
# Constants
STEP = .8   # size of a step of a lattice path
BUFF = .4   # fraction of step that is buffer for bounding box 
def slide_title(text):
    return text.to_corner(UP + LEFT).shift(.2*UP + .2*LEFT)

# GENERATORS

def lattice_paths(m, n):
    # Returns all the lattice paths from (0,0) to (m,n), as generator.

    if m == 0:
        yield [1]*n
    elif n == 0:
        yield [0]*m
    else:
        for p in lattice_paths(m-1, n):
            yield p + [0]
        for p in lattice_paths(m, n-1):
            yield p + [1]

def dyckpaths(n, k, level=0, started=False):
    # n is twice the size, k is the number of peaks, level is the distance from the main diagonal,
    # started is True if we already started a vertical segment, False otherwise.
    # returns the set of Dyck paths as lists of 0's and 1's.

    if level >= 0:
        if n == 0:
            if level == 0 and k == 0:
                yield []
        else:
            if started:  # The previous step is vertical.
                for p in dyckpaths(n-1, k, level=level+1, started=True):
                    yield [1] + p
                for p in dyckpaths(n-1, k, level=level-1, started=False):
                    yield [0] + p
            else:  # The previous step is horizontal.
                if k > 0:
                    for p in dyckpaths(n-1, k-1, level=level+1, started=True):
                        yield [1] + p
                for p in dyckpaths(n-1, k, level=level-1, started=False):
                    yield [0] + p

def paths(n, m=0, dyck=False, labels=None, drises=0, dvalleys=0):
    # Draws all the Dyck paths, or Square paths ending East, of size n,
    # with m labels equal to 0, drises decorated rises and dvalleys decorated valleys.
    # labels is the composition of the multiplicities of the labels. None for the parking version 1,...,n

    # If dyck is True, only builds Dyck paths.
    if dyck:
        paths = [Path(p) for j in range(n) for p in dyckpaths(2*(n), j+1)]

    # If dyck is False, computes square paths ending East.
    else:
        paths = [Path(p+[0]) for p in lattice_paths(n-1, n)]

    # Sets the deafult set of labels to [n].
    if labels is None:
        labels = [m] + [1]*(n-m)

    if labels is False:
        decorated_paths = [Path(p.path, labels=None, rises=list(r), valleys=list(v))
                           for p in paths
                           for r, v in product(combinations(p.findrises(), drises),
                                               combinations(p.findvalleys(), dvalleys))
                           ]
        return decorated_paths
    else:
        decorated_labelled_paths = [Path(p.path, labels=l, rises=list(r), valleys=list(v))
                                    for p in paths
                                    for l in p.labellings(labels)
                                    for r, v in product(combinations(p.findrises(), drises),
                                                        combinations(Path(p.path, l).findvalleys(), dvalleys))
                                    ]
        return decorated_labelled_paths

def polyominoes(m, n, labels=None, reduced=False):
    # m is the width, n is the height

    polyominoes = []

    for d in dyckpaths(2*(m+n-1), m):

        redpath = []
        greenpath = []

        for i in range(2*(m+n-1)):
            if d[i] == 1:
                if i == 0:
                    redpath += [1]
                else:
                    redpath += [d[i-1]]
            else:
                greenpath += [1-d[i-1]]

        redpath += [0]
        greenpath += [1]

        polyominoes += [Polyomino(redpath, greenpath)]

    if labels is None:
        labels = [0] + [1]*(m+n-1)

    if labels is False:
        return polyominoes
    else:
        return [Polyomino(p.redpath, p.greenpath, labelling) for p in polyominoes for labelling in p.labellings(labels)]

def mu_labellings(blocks, labels, strict=True, reverse=False):
    if len(blocks) == 0:
        yield []
    else:
        if strict == True:
            iter = combinations(set(labels), blocks[0])
        else:
            iter = distinct_combinations(labels, blocks[0])
        for block in iter:
            xlist = list(block)
            xlist.sort(reverse=reverse)
            for xlabels in mu_labellings(blocks[1:], list((multiset(labels) - multiset(block))), strict=strict, reverse=reverse):
                yield xlist + xlabels

# CLASSES

class AreaWord(object):
    # Defines the AreaWord object.

    def __init__(self, word):

        # It's a list of integers, that should be a Dyck word.
        self.word = word
        # It's the length of the area word.
        self.length = self.getlength()

    def getlength(self):
        # Computes the length once. It's stored during the initialization.
        return len(self.word)

    def to_path(self):
        # Converts the area word into a path using the canonical correspondence.

        # Initialise the path.
        path = [0]*(-self.word[0])
        # Define an extended word so that the Dyck path goes back to the diagonal at the end.
        xword = self.word + [0]

        for i in range(self.length):
            # For each letter of the area word, it adds a vertical step to the path, followed by a number
            # of horizontal steps equal to the difference with the next letter, plus one.
            path += [1] + [0]*(xword[i] - xword[i+1] + 1)

        return Path(path)

class Path(object):
    # Defines the Path object.

    def __init__(self, path, labels=None, rises=[], valleys=[]):

        # It's the actual path, stored as a string of 0's (east steps) and 1's (north steps)
        self.path = path
        # It's the list of the labels of the path, to read bottom to top. Default is None.
        self.labels = labels
        # These are the indices of the decorated rises.
        self.rises = rises
        # These are the indices of the decorated valleys.
        self.valleys = valleys

        # It's the size of the path, which is half the number of steps.
        self.size = self.get_size()

        self.aword = self.area_word().word
        # It's the disance between the main diagonal and the base diagonal.
        self.shift = - min(self.aword)

        # The vector for shifting Mobjects to center
        self.to_center = array([-0.5*self.size, -0.5*self.size,0])
        # Bounding box around pictures, necessary to include in each component to ensure exact layering.
        self.bounding_box = self.draw_bounding_box()

        

    # animation functions

    def draw_bounding_box(self, opacity = 0):
        return Rectangle(width = self.size + 2*BUFF, height= self.size + 2*BUFF).set_stroke(opacity = opacity)

    def draw_grid(self):
        grid = []
        for i in range(self.size + 1):
            grid += Line(array([i,0,0]),array([i,self.size,0]))
            grid += Line(array([0,i,0]), array([self.size, i, 0]))
        
        grid = VGroup(*grid)

        grid.shift(self.to_center)
        bb = deepcopy(self.bounding_box)
        grid = VGroup(grid, bb)
        grid.scale(STEP)

        return grid        

    def draw_path(self):
        path = []
        point = ORIGIN
        for i in self.path:
            if i == 0:
                newpoint = point + RIGHT
            else:
                newpoint = point + UP
            path += Line(point, newpoint, color = PINK, width = 6)
            point = newpoint
        out = VGroup(*path)

        out.shift(self.to_center)
        bb = deepcopy(self.bounding_box)
        out = VGroup(out, bb)
        out.scale(STEP)

        return out

    def draw_labels(self):
        out = []
        for i in range(self.size):
                out += Tex(f"{self.labels[i]}").shift((i + .5)*UP + (i - self.aword[i] + .5)*RIGHT)

        out = VGroup(*out)

        out.shift(self.to_center)
        bb = deepcopy(self.bounding_box)
        out = VGroup(out, bb)
        out.scale(STEP)
        return out
    
    def draw_decorations(self):
        out = []
        for i in self.rises:
            out += MathTex(r"\ast").shift((i+.5)*UP + (i - self.aword[i] - .5)*RIGHT)
        for i in self.valleys:
            out += MathTex(r"\bullet").shift((i+.5)*UP + (i - self.aword[i] - .5)*RIGHT)
        
        out = VGroup(*out)

        out.shift(self.to_center)
        bb = deepcopy(self.bounding_box)
        out = VGroup(out, bb)
        out.scale(STEP)
        return out

    def draw_diagonal(self, diag=0):
        # draw the line y = x + diag in the grid
        out= Line(
            array([-min(0,diag), max(diag,0),0]),
            array([self.size - max(diag,0), self.size + min(0,diag),0])
        ).set_opacity(.4)
        out.shift(self.to_center)
        bb = deepcopy(self.bounding_box)
        out = VGroup(out, bb)
        out.scale(STEP)

        return out

    def draw(self, diagonal = False):
        out = VGroup(
            self.draw_grid(), 
            self.draw_path(),
            self.draw_labels(),
            self.draw_decorations())
        if diagonal: 
            out = VGroup(
                out,
                self.draw_diagonal(-self.shift)
            )
        return out
 
    def circle_labels(self, labs):
        # i in labs -> circe label in i-th row
        out = []
        for i in labs:
                out += Circle(color = WHITE, radius = .4).shift((i - .5)*UP + (i - self.aword[i-1]  - .5)*RIGHT)

        out = VGroup(*out)

        out.shift(self.to_center)
        bb = deepcopy(self.bounding_box)
        out = VGroup(out, bb)
        out.scale(STEP)
        return out

    def highlight_squares(self, squares):
        # (i,j) in squares: i-th column, j-th row
        out = []
        for (i,j) in squares:
            out += Square(side_length = 1, color = PINK).set_opacity(.3).set_stroke(width = 0).shift((i-.5)*RIGHT + (j-.5)*UP)
        
        out = VGroup(*out)

        out.shift(self.to_center)
        bb = deepcopy(self.bounding_box)
        out = VGroup(out, bb)
        out.scale(STEP)
        return out

    def highlight_diagonal(self, diag):
        squares = []
        for k in range(self.size - abs(diag)):
            squares += [(-min(0,diag) + k + 1,max(diag,0) + k + 1)]

        out = self.highlight_squares(squares)
     
        return out

    def higlight_area(self):
        # highlights the squares contributing to the area 
        out = []
        aw = [a + self.shift for a in self.area_word().word]
        for j,a in enumerate(aw):
            if j not in self.rises:
                for i in range(a):
                    out += [(j-i + self.shift, j + 1)]
        return self.highlight_squares(out)

    def higlight_steps(self, steps):
        path = []
        point = ORIGIN
        for index, i in enumerate(self.path):
            if i == 0:
                newpoint = point + RIGHT
            else:
                newpoint = point + UP
            if index+1 in steps:
                path += Line(point, newpoint, color = GREEN, width = 6)
            point = newpoint
        out = VGroup(*path)

        out.shift(self.to_center)
        bb = deepcopy(self.bounding_box)
        out = VGroup(out, bb)
        out.scale(STEP)

        return out

    def monomial_string(self, qstat = "dinv", tstat = "area"):
        # the tex mobject that is q^qstat(p) t^tstat(p) x^p
        string = ""

        if getattr(self, qstat)() == 1:
            string += "q"
        elif getattr(self, qstat)() > 1:
            string += "q^{%d}" % getattr(self, qstat)()
        
        if getattr(self, tstat)() == 1:
            string += "t"
        elif getattr(self, tstat)() > 1:
            string += "t^{%d}" % getattr(self, tstat)()
        
        for i in range(max(self.labels)):
            if self.labels.count(i+1) == 1:
                string += "x_{%d}" %(i+1)
            if self.labels.count(i+1) >1:
                string += "x_{%d}^{%d}" %(i+1,self.labels.count(i+1))

        return string

    # Math functions

    def get_size(self):
        # Computes the length of the Dyck path.
        return int(len(self.path) / 2)

    def area_word(self):
        # Returns the area word of the path, as AreaWord.

        areaword = []  # Initializes the area word to an empty string.
        level = 0  # Sets starting level to 0.

        for i in self.path:
            if i == 1:  # If the Dyck path has a vertical step, it adds a letter to the area word, and then goes up a level.
                areaword += [level]
                level += 1
            elif i == 0:  # If the Dyck path has a horizontal step, it goes down a level.
                level -= 1
            else:
                raise ValueError('Entries of the path must be 0 or 1.')

        return AreaWord(areaword)

    def labellings(self, composition=None):
        # Returns all the possible labellings of the path, provided that it has no labels and no decorations.
        # It is possible to specify which labels to use, composition[i] being the number of i's appearing.

        # The deafult set of labels is [n].
        if composition is None:
            composition = [0] + [1]*self.size
        assert(self.size == sum(composition)), 'The number of labels does not match the size of the path.'

        # Find the composition given by the vertical steps of the path.
        peaks = sorted(set(multiset([sum(self.path[:i]) for i in range(2*self.size+1)])
                           - multiset(range(1, self.size+1))))
        blocks = [peaks[i+1] - peaks[i] for i in range(len(peaks)-1)]

        # Define the set of labels.
        labels = [x for y in [[i]*composition[i] for i in range(len(composition))] for x in y]
        labellings = [labelling for labelling in mu_labellings(blocks, labels) if not (
            (self.aword[0] == 0 and labelling[0] == 0)
            or (len([i for i in range(self.size) if self.aword[i] == -self.shift and labelling[i] > 0]) == 0)
        )]

        return labellings

    def findrises(self):
        # Returns the indices of all the double rises of the path.
        return [i for i in range(1, self.size) if self.aword[i] > self.aword[i-1]]

    def findvalleys(self, contractible=True):
        # Returns the indices of all the (contractible) valleys of the path.
        assert self.valleys == [], 'The path already has some decorated valleys.'

        return [i for i in range(self.size)
                if (i == 0 and self.aword[i] < -1)
                or (i == 0 and self.aword[i] == -1 and (self.labels is None or self.labels[i] > 0))
                or (i > 0 and self.aword[i] < self.aword[i-1])
                or (i > 0 and self.aword[i] == self.aword[i-1]
                    and (contractible == False or self.labels is None or self.labels[i-1] < self.labels[i]))]

    def dinv(self):
        # Returns the dinv. If the path is labelled, it takes the labelling into account.

        dinv = 0  # Initializes dinv to 0.

        # Goes through the letters of the area word.
        for i in range(self.size):
            # Bonus dinv
            if self.aword[i] < 0 and (self.labels is None or self.labels[i] > 0):
                dinv += 1
            if i not in self.valleys:  # Skip decorated valleys
                for j in range(self.size)[i+1:]:  # Looks at the right.
                    if self.aword[j] == self.aword[i]-1:  # Secondary dinv
                        # Checks labels
                        if self.labels is None or self.labels[j] < self.labels[i]:
                            dinv += 1
                    elif self.aword[j] == self.aword[i]:  # Primary dinv
                        # Checks labels
                        if self.labels is None or self.labels[j] > self.labels[i]:
                            dinv += 1
            else:  # elif self.aword[i] > 0 or self.labels[i] > 0: # Subtract 1 for each decorated valley
                dinv += -1

        return dinv

    def area(self):
        # Returns the area. Ignores rows with decorated rises.

        if self.rises is not []:
            area = sum(self.aword[i] + self.shift for i in range(self.size) if i not in self.rises)
        else:
            area = sum(self.aword[i] + self.shift for i in range(self.size))
        return area

    def pmaj(self):
        # If the path is a Dyck path, returns the pmaj. Otherwise, returns -1.

        if self.shift > 0:
            return -1
        else:
            return maj(self.parking_word()[::-1])

    def zero(self):
        return 0

    def column(self, i):
        # Returns the index of the column (numbered starting from 0) containing the label with index i.
        return i - self.aword[i] + self.shift

    def dinv_contribution(self, i, primary='left', secondary='right', ignore_valleys=True):
        # Computes the contribution to the dinv of a given index i.

        dinv = 0

        if primary == 'left':
            for j in range(i):
                if self.aword[j] == self.aword[i] and self.labels[j] < self.labels[i] and j not in self.valleys:
                    dinv += 1
        elif primary == 'right' and i not in self.valleys:
            for j in range(self.size)[i+1:]:
                if self.aword[j] == self.aword[i] and self.labels[j] > self.labels[i] and (j not in self.valleys or ignore_valleys == False):
                    dinv += 1

        if secondary == 'left':
            for j in range(i):
                if self.aword[j] == self.aword[i]+1 and self.labels[j] > self.labels[i] and j not in self.valleys:
                    dinv += 1
        elif secondary == 'right' and i not in self.valleys:
            for j in range(self.size)[i+1:]:
                if self.aword[j] == self.aword[i]-1 and self.labels[j] < self.labels[i] and (j not in self.valleys or ignore_valleys == False):
                    dinv += 1

        if self.aword[i] < 0:
            dinv += 1

        if i in self.valleys:
            dinv -= 1

        return dinv

    def diagonals(self):
        # Computes the list whose entries are the labels appearing in each diagonal, bottom to top.

        if self.labels is None:
            raise AttributeError('The path is not labelled.')
        else:
            diagonals = [[]]*(self.shift + max(self.aword) + 1)

            for i in range(self.size):
                diagonals[self.aword[i] + self.shift] = diagonals[self.aword[i] +
                                                                  self.shift] + [self.labels[i]]

            return diagonals

    def diagonal_word(self):
        # Returns the word obtained by sorting the diagonals in decreasing order, bottom to top.

        if self.labels is None:
            return [x+1 for x in range(self.size)]
        else:
            return [x for d in self.diagonals() for x in sorted(d)[::-1]]

    def parking_word(self):
        # Returns the parking word of the path, i.e. the word whose descent set of the inverse gives the pmaj.

        if self.shift > 0:
            raise NotImplementedError(
                'We do not know how to compute the parking word for square paths.')
        else:
            if self.labels is None:
                labels = [x+1 for x in range(self.size)]
            else:
                labels = self.labels

            stack = multiset()  # stack is the (multi)set of unused labels.
            parkword = []  # Initializes the parking word to an empty string.
            # x is the horizontal coordinate, y is the vertical coordinate.
            x = y = 0
            v = []

            for i in range(2*self.size):
                if self.path[i] == 1:
                    # If we read a vertical step, we add the corresponding label to the unused ones, then we increase y by one.
                    stack += {labels[y]}
                    y += 1
                elif i < 2*self.size-1 and self.path[i] == 0 and self.path[i+1] == 1 and y in self.valleys:
                    # If we read a decorated valley, we annotate when to read it.
                    v += [self.aword[y]]
                else:
                    # If we read a horizontal step that is not a decorated valley,
                    # we make a note to read it immediately.
                    v += [0]

                    for w in v:
                        if w == 0:
                            # For each horizontal step we are supposed to have,

                            if parkword == [] or (stack & set(range(parkword[x-1]+1)) == set()):
                                # we take the highest unused label smaller than the previous one (if any),
                                u = max(stack)
                            else:
                                # or else we just take the highest one.
                                u = max(stack & set(range(parkword[x-1]+1)))

                            # We add the label to the pmaj reading word,
                            parkword += [u]
                            stack -= {u}  # remove it from the unused labels,
                            x += 1  # then increase x by one.

                    # Finally we update the valley counters.
                    v = [w-1 for w in v if w > 0]

            return parkword

    def reading_word(self, read='standard'):
        # Computes the reading word of a path, bottom to top.

        if read == 'standard' or read == 'diagonal':
            # Reading word according to the dinv statistic, i.e. along diagonals, left to right, bottom to top.
            return [x for d in self.diagonals() for x in d]
        elif read == 'vertical':
            # Reading word according to the pmaj statistic, i.e. along columns, left to right, bottom to top.
            return self.labels
        else:
            raise AttributeError('Reading order not recognized.')

    def diagonal_composition(self):
        # Computes the diagonal composition of the path, ignoring decorated rises and valleys.

        aux_aword = [self.aword[i] for i in range(
            self.size) if i not in self.rises and i not in self.valleys] + [-self.shift]
        diagonal_indices = [i for i in range(
            len(aux_aword)) if aux_aword[i] == -self.shift]
        composition = [diagonal_indices[i+1] - diagonal_indices[i]
                       for i in range(len(diagonal_indices)-1)]

        return composition

    def gessel(self, read='standard'):
        ls = [x for x in self.reading_word(read)[::-1] if x > 0]
        return set_to_composition(ides(ls), len(ls))

class Polyomino(object):

    def __init__(self, redpath, greenpath, labels=None, reduced=False):
        # Defines the Polyomino object.

        self.redpath = redpath  # It's the red (top) path of the polyomino.
        # It's the green (bottom) path of the polyomino.
        self.greenpath = greenpath
        # True if the polyomino is reduced, false otherwise.
        self.reduced = reduced
        # The set of labels of the polyomino, read left to right, bottom to top.
        self.labels = labels

        # Returns m+n, namely the total length of the paths.
        self.length = self.getlength()
        # Returns n, which is the height of the polyomino.
        self.height = self.getheight()
        # Returns m, which is the width of the polyomino.
        self.width = self.getwidth()
        # Returns the polyomino as a collection of cells.
        self.cells = self.getcells()
        
        # The vector for shifting Mobjects to center
        self.to_center = array([-0.5*self.width, -0.5*self.height,0])
        # Bounding box around pictures, necessary to include in each component to ensure exact layering.
        self.bounding_box = self.draw_bounding_box()


    # animation functions

    def draw_bounding_box(self, opacity = 0):
        return Rectangle(width = self.width + 2*BUFF, height= self.height + 2*BUFF).set_stroke(opacity = opacity)

    def draw_grid(self):
        
        grid = []
        for i in range(self.width + 1):
            grid += Line(array([i,0,0]),array([i,self.height,0]))
        for i in range(self.height + 1):
            grid += Line(array([0,i,0]), array([self.width, i, 0]))
        
        grid = VGroup(*grid)

        grid.shift(self.to_center)
        bb = deepcopy(self.bounding_box)
        grid = VGroup(grid, bb)
        grid.scale(STEP)

        return grid

    def draw_redpath(self):
        path = []
        point = ORIGIN
        for i in self.redpath:
            if i == 0:
                newpoint = point + RIGHT
            else:
                newpoint = point + UP
            path += Line(point, newpoint, color = PINK, width = 5)
            point = newpoint
        out = VGroup(*path)

        out.shift(self.to_center)
        bb = deepcopy(self.bounding_box)
        out = VGroup(out, bb)
        out.scale(STEP)

        return out

    def draw_greenpath(self):
        path = []
        point = ORIGIN
        for i in self.greenpath:
            if i == 0:
                newpoint = point + RIGHT
            else:
                newpoint = point + UP
            path += Line(point, newpoint, color = GREEN, width = 5)
            point = newpoint
        out = VGroup(*path)

        out.shift(self.to_center)
        bb = deepcopy(self.bounding_box)
        out = VGroup(out, bb)
        out.scale(STEP)

        return out
    
    def draw_paths(self):
        return VGroup(self.draw_redpath(), self.draw_greenpath())

    def draw_labels(self):
        out = []
        for i in range(self.width):
                for j in range(self.height):
                    if self.labels[i][j] != None:
                        out += Tex(f"{self.labels[i][j]}").shift((i + .5)*RIGHT + (j + .5)*UP)
        out = VGroup(*out)

        out.shift(self.to_center)
        bb = deepcopy(self.bounding_box)
        out = VGroup(out, bb)
        out.scale(STEP)
        return out

    def draw(self):
        return VGroup(
            self.draw_grid(), 
            self.draw_paths(),
            self.draw_labels())

    def circle_labels(self, labs):
        # (i,j) in labs -> circle label in i-th col and j-th row   
        out = []
        for (i,j) in labs:
            out += Circle(color = WHITE, radius = .4).shift((i - .5)*RIGHT + (j - .5)*UP)
        
        out = VGroup(*out)

        out.shift(self.to_center)
        bb = deepcopy(self.bounding_box)
        out = VGroup(out, bb)
        out.scale(STEP)
        return out
    
    def highlight_squares(self, squares):
        # (i,j) in squares: i-th column, j-th row
        out = []
        for (i,j) in squares:
            out += Square(side_length = 1, color = WHITE).set_opacity(.3).set_stroke(width = 0).shift((i-.5)*RIGHT + (j-.5)*UP)
        
        out = VGroup(*out)

        out.shift(self.to_center)
        bb = deepcopy(self.bounding_box)
        out = VGroup(out, bb)
        out.scale(STEP)
        return out
    
    def highlight_diagonal(self, diag, color = PINK):
        squares = []
        y = min(diag, self.height)
        x = 1 + max(0, diag - y)
        while True:
            print("(x,y) = ", (x,y))
            squares += [(x,y)]
            x += 1
            y -= 1
            if x > self.width or y < 1:
                break
        
        out = self.highlight_squares(squares)
     
        return out

    # Math functions

    def getlength(self):
        # Computes the length once. It's stored during the initialization.
        if self.reduced:
            return len(self.redpath) + 2
        else:
            return len(self.redpath)

    def getheight(self):
        # Computes the height once. It's stored during the initialization.

        if self.reduced:
            return sum(self.redpath) + 1
        else:
            return sum(self.redpath)

    def getwidth(self):
        # Computes the width once. It's stored during the initialization.
        return self.length - self.height

    def getcells(self):
        # Returns the array of the cells of the polyomino.
        # Outer cells contain -1, border cells contain 0, inner cells contain 1.

        if self.reduced:
            return self.expand().getcells()

        # Initalise every cell to -1.
        cells = array([[-1]*self.height]*self.width)

        for j in range(self.height):
            # Position of the vertical red step in the j-th row.
            i1 = get_nth_index(self.redpath, 1, j+1)
            # Position of the vertical green step in the j-th row.
            i2 = get_nth_index(self.greenpath, 1, j+1)
            for i in range(i1-j, i2-j):
                if i > 0 and j > 0 and cells[i-1][j] > -1 and cells[i][j-1] > -1:
                    # Inner cell.
                    cells[i][j] = 1
                else:
                    # Border cell.
                    cells[i][j] = 0

        return cells

    def labellings(self, composition=None):
        # Returns all the possible labellings of the polyomino.
        # It is possible to specify which labels to use, composition[i] being the number of i's appearing.

        if self.reduced:
            return self.expand().labellings(composition=composition)

        # The deafult set of labels is [m+n-1].
        if composition is None:
            composition = [0] + [1]*(self.length-1)

        # Define the set of labels.
        labels = [x for y in [[i]*composition[i]
                              for i in range(len(composition))] for x in y]

        labellings = []
        row_composition = [len([x for x in self.cells[:, j] if x == 0])
                           for j in range(self.height)]

        for xlabelling in mu_labellings(row_composition, labels, reverse=True):
            # Initalise every cell to None.
            labelling = array([[None]*self.height]*self.width)

            h = 0
            is_valid = True
            for j in range(self.height):
                for i in range(self.width):
                    if self.cells[i, j] == 0:
                        # The cell is a border cell
                        if xlabelling[h] > max([-1] + [x for x in labelling[i, :] if x is not None]):
                            # The labelling is strictly increasing along columns
                            # -1 is artificially added to avoid empty sequences
                            labelling[i, j] = xlabelling[h]
                            h += 1
                        else:
                            is_valid = False

                    if not is_valid:
                        break
                if not is_valid:
                    break

            if is_valid:
                labellings += [labelling]

        return labellings

    def expand(self):
        if self.reduced:
            return Polyomino([1] + self.redpath + [0], [0] + self.greenpath + [1], self.labels, reduced=False)
        else:
            return self

    def shrink(self):
        if self.reduced:
            return self
        else:
            return Polyomino(self.redpath[1:self.length-1], self.greenpath[1:self.length-1], self.labels, reduced=True)

    def area(self):
        # Computes the area.
        area = 0
        for j in range(self.height):
            for i in range(self.width):
                if self.cells[i, j] == 1:
                    if self.labels is None or min([x for x in self.labels[:i, j] if x is not None]) > max([x for x in self.labels[i, :j] if x is not None]):
                        area += 1
        return area

    def bounceword(self):
        # Returns the bounce word of a polyomino, as list.

        if not self.reduced:
            i = j = 1  # i is for the red path, j is for the green path.
            l = 1  # l is the current label.
            # Initializes the bounce word to the list [0,1]. All the bounce words of polyominoes start like that.
            bounceword = [0, 1]

            while j - sum(self.greenpath[:j]) + sum(self.redpath[:i]) < self.length:
                # The number of horizontal steps of the green path, plus the number of vertical steps of the red path, is not equal to m+n yet.

                while i < min(self.length, j - sum(self.greenpath[:j]) + sum(self.redpath[:i])):
                    # The bounce path goes vertically until it hits the end of a horizontal red step.
                    if self.redpath[i] == 1:
                        bounceword += [l]
                    i += 1
                l += 1

                while j < self.length and sum(self.greenpath[:j]) < sum(self.redpath[:i]):
                    # The bounce path goes horizontally until it hits the end of a vertical green step.
                    if self.greenpath[j] == 0:
                        bounceword += [l]
                    j += 1
                l += 1

            return bounceword

        else:
            return [i-2 for i in Polyomino([1] + [1-x for x in self.greenpath] + [0], [0] + [1-x for x in self.redpath] + [1]).bounceword()[2:]]

    def bounce(self):
        # Computes the bounce as a sum of the letters of the bounce word, with the same rule of the area.

        # bounceword = self.bounceword()
        # bounceword = [x-2 for x in self.bounceword()[1:] if x % 2 == 0]
        # bounce = int(sum(((bounceword[i]+1) // 2) for i in range(len(bounceword))))

        vbounce = [x-1 for x in self.bounceword() if x % 2 == 1]
        hbounce = [x-1 for x in self.bounceword() if x % 2 == 0]
        bounce = 0

        for i in range(1, self.width):
            # if min([x for x in self.labels[i,:] if x is not None]) > 0:
            bounce += int(hbounce[i] // 2)

        # m = max(self.reading_word())
        for j in range(self.height):
            # if max([x for x in self.labels[:,j] if x is not None]) < m:
            bounce += int(vbounce[j] // 2)

        return bounce

    def zero(self):
        return 0

    def pmaj(self):
        return self.sh_pmaj()

    def sh_parking_word(self):

        x = 0
        j = 0
        L = multiset()
        pword = []

        for i in range(self.length-1):

            if self.greenpath[i] == 0:
                L += multiset([z for z in self.labels[x, :] if z is not None])
                x += 1

            if min(L) == 0:
                L -= {0}
                j += 1
            else:
                if pword == [] or (L & set(range(pword[i-j-1]+1)) == set()):
                    # we take the highest unused label smaller than the previous one (if any),
                    u = max(L)
                else:
                    # or else we just take the highest one.
                    u = max(L & set(range(pword[i-j-1]+1)))

                pword += [u]  # We add the label to the pmaj reading word,
                L -= {u}  # remove it from the unused labels.

        return pword

    def sh_pmaj(self):
        return maj(self.sh_parking_word()[::-1])

    def double_pmaj(self):

        x = 0
        y = 0

        R = multiset()
        G = multiset()

        redword = []
        greenword = []

        for i in range(self.length-1):

            if self.greenpath[i] == 0:
                R += multiset([z for z in self.labels[x, :] if z is not None])

                if i > 0:
                    R -= {min(multiset([z for z in self.labels[x, :] if z is not None]))}

                x += 1

            if i == 0 or self.greenpath[i] == 1:
                if redword == [] or (R & set(range(redword[-1]+1)) == set()):
                    # we take the highest unused label smaller than the previous one (if any),
                    u = max(R)
                else:
                    # or else we just take the highest one.
                    u = max(R & set(range(redword[-1]+1)))

                redword += [u]  # We add the label to the pmaj reading word,
                R -= {u}  # remove it from the unused labels.

            if self.redpath[i] == 1:
                G += multiset([z for z in self.labels[:, y] if z is not None])
                G -= {max(multiset([z for z in self.labels[:, y] if z is not None]))}

                y += 1

            if self.redpath[i] == 0:
                if greenword == [] or (G & set(range(greenword[-1]+1, self.length)) == set()):
                    # we take the smallest unused label greater than the previous one (if any),
                    u = min(G)
                else:
                    # or else we just take the highest one.
                    u = min(G & set(range(greenword[-1]+1, self.length)))

                greenword += [u]  # We add the label to the pmaj reading word,
                G -= {u}  # remove it from the unused labels.

        return maj(redword[::-1]) + sum(asc(greenword[::-1]))

    def to_zero(self):
        # Convert an unlabelled polyomino into a zero area, zero pmaj labelled polyomino.

        redpath = self.redpath
        greenpath = [0] + self.redpath[1:self.length-1] + [1]
        # Initalise every cell to None.
        labels = array([[None]*self.height]*self.width)

        L = multiset()
        x = self.width
        y = self.height-1
        i0 = 0

        for i in range(self.length-1):
            if self.redpath[::-1][i] == 0:

                x -= 1
                i1 = get_nth_index(self.greenpath[::-1], 0, self.width - x)
                L += multiset(range(i0+1, i1+1))

                labels[x, y] = max(L)
                L -= {max(L)}
                i0 = i1

            elif self.redpath[::-1][i] == 1:

                y -= 1
                labels[x, y] = max(L)
                L -= {max(L)}

        return(Polyomino(redpath, greenpath, labels=labels))

    def rotate(self):
        m = self.width
        n = self.height
        labels = array([[None]*n]*m)  # Initalise every cell to None.
        for i in range(m):
            for j in range(n):
                if self.labels[m-i-1, n-j-1] is not None:
                    labels[i, j] = m+n-self.labels[m-i-1, n-j-1]
        return Polyomino(self.greenpath[::-1], self.redpath[::-1], labels=labels)

    def reading_word(self, read='standard'):

        if read == 'rows':
            word = []
            for j in range(self.height):
                word += [x for x in self.labels[:, j][::-1] if x is not None]

        elif read == 'columns':
            word = []
            for i in range(self.width):
                word += [x for x in self.labels[self.width-i-1, :]
                         if x is not None]

        elif read == 'standard' or read == 'clockwise':
            word = []
            for j in range(self.height):
                row = [x for x in self.labels[:, self.height-j-1]
                       [::-1] if x is not None]
                if len(row) > 1:
                    word += [x for x in row if x != max(row)]

            word += [x for x in self.labels[0, :] if x is not None]

            for i in range(1, self.width):
                column = [x for x in self.labels[i, :] if x is not None]
                if len(column) > 1:
                    word += [x for x in column if x != min(column)]

        elif read == 'diagonal':
            word = []
            for i in range(-self.width + 1, self.height):
                for j in range(min([i+self.width, self.width, self.height, self.height-i])):
                    y = max([j, i + j])
                    x = y - i
                    if self.labels[x, y] is not None:
                        word += [self.labels[x, y]]

        elif read == 'rank':
            word = []
            keys = sorted([[i, j] for i in range(self.width) for j in range(self.height)],
                          key=lambda cell: self.width*cell[1] - self.height*cell[0] + int(floor(gcd(self.width, self.height)/self.width)*cell[0]/self.width))

            for c in keys:
                if self.labels[c[0], c[1]] is not None:
                    word += [self.labels[c[0], c[1]]]
        else:
            word = self.labels

        return word

    def gessel(self, read='standard'):
        # return self.to_labelled_dyckpath().gessel(read)
        ls = [x for x in self.reading_word(read)[::-1] if x > 0]
        return set_to_composition(ides(ls), len(ls))

class YT(object):
    def __init__(self, partition, labels = None):
        self.partition = partition
        self.labels = labels

    def draw(self):
        out = VGroup()
        out2 = VGroup()
        k =0
        for i in range(len(self.partition)):
            for j in range(self.partition[i]):
                out.add(Square(side_length=1).move_to(i*UP + j*RIGHT))
                if self.labels != None:
                    lab = MathTex(f'{self.labels[k]}').move_to(i*UP + j*RIGHT)
                    out.add(lab)
                    k += 1
        out.shift(-out.get_center())  
        return out