from collections import defaultdict


# Class to represent a graph
class Graph:

    def __init__(self, vertices):
        self.V = vertices  # No. of vertices
        self.graph = []  # default dictionary
        self.result = []
        self.parent = []
        self.rank = []
        # to store graph

    # function to add an edge to graph
    def add_edge(self, u, v, weight):
        self.graph.append([u, v, weight])

        # A utility function to find set of an element i

    # (uses path compression technique)
    def find(self, parent, i):
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])

        # A function that does union of two sets of x and y

    # (uses union by rank)
    def union(self, parent, rank, x, y):
        x_root = self.find(parent, x)
        y_root = self.find(parent, y)

        # Attach smaller rank tree under root of
        # high rank tree (Union by Rank)
        if rank[x_root] < rank[y_root]:
            parent[x_root] = y_root
        elif rank[x_root] > rank[y_root]:
            parent[y_root] = x_root

            # If ranks are same, then make one as root
        # and increment its rank by one
        else:
            # Sets the parent of y_root (v) to be x_root (u)
            parent[y_root] = x_root
            rank[x_root] += 1

    # The main function to construct MST using Kruskal's
    # algorithm
    def KruskalMST(self):

        # result = []  # This will store the resultant MST

        i = 0  # An index variable, used for sorted edges
        e = 0  # An index variable, used for result[]

        # Step 1:  Sort all the edges in non-decreasing
        # order of their
        # weight.  If we are not allowed to change the
        # given graph, we can create a copy of graph
        self.graph = sorted(self.graph, key=lambda item: item[2])

        # parent = []
        # rank = []

        # Create V subsets with single elements
        for node in range(self.V):
            self.parent.append(node)
            self.rank.append(0)

            # Number of edges to be taken is equal to V-1
        while e < self.V - 1:

            # Step 2: Pick the smallest edge and increment
            # the index for next iteration
            """
                We get the first edge
            """
            u, v, w = self.graph[i]
            i = i + 1  # TODO find out what this does
            x = self.find(self.parent, u)  # Finds the vertex from where this edge starts
            y = self.find(self.parent, v)  # Finds the vertex to which this edge ends

            # If including this edge doesn't cause cycle,
            # include it in result and increment the index
            # of result for next edge
            """
                
            """
            if x != y:
                e = e + 1
                self.result.append([u, v, w])
                self.union(self.parent, self.rank, x, y)
                # Else discard the edge

        # print the contents of result[] to display the built MST
        # print ("Following are the edges in the constructed MST")
        # for u, v, weight in result:
        #     # print str(u) + " -- " + str(v) + " == " + str(weight)
        #     print("%d -- %d == %d" % (u, v, weight))
        # print(parent)
        # # Driver code


def compare_contents(g_by_random, g_by_vertex_order):
        for i in range(len(g_by_random)):
            u1, v1, weight1 = g_by_random[i]
            u2, v2, weight2 = g_by_vertex_order[i]
            print("Random order:", "%d -- %d == %d" % (u1, v1, weight1), "VS Vertex order:", "%d -- %d == %d" % (u2, v2, weight2))


def by_random(g):
    g.add_edge(0, 1, 23)
    g.add_edge(0, 3, 7)
    g.add_edge(0, 6, 11)
    g.add_edge(0, 4, 1)
    g.add_edge(4, 5, 2)
    g.add_edge(4, 8, 19)
    g.add_edge(3, 8, 5)
    g.add_edge(3, 6, 15)
    g.add_edge(1, 12, 7)
    g.add_edge(1, 7, 4)
    g.add_edge(12, 10, 22)
    g.add_edge(7, 11, 12)
    g.add_edge(7, 6, 10)
    g.add_edge(11, 10, 9)
    g.add_edge(10, 9, 11)
    g.add_edge(6, 10, 18)
    g.add_edge(6, 9, 13)
    g.add_edge(6, 2, 6)


def by_vertex_order(g):
    g.add_edge(0, 1, 23)
    g.add_edge(0, 3, 7)
    g.add_edge(0, 6, 11)
    g.add_edge(0, 4, 1)
    g.add_edge(1, 12, 7)
    g.add_edge(1, 7, 4)
    # 6
    g.add_edge(2, 6, 6)
    g.add_edge(3, 8, 5)
    g.add_edge(3, 6, 15)
    # 9
    g.add_edge(4, 5, 2)
    g.add_edge(4, 8, 19)
    # 11
    g.add_edge(6, 7, 10)
    g.add_edge(6, 9, 13)
    g.add_edge(6, 10, 18)
    # 14
    g.add_edge(7, 11, 12)
    g.add_edge(9, 10, 11)
    g.add_edge(10, 11, 9)
    g.add_edge(10, 12, 22)
    # 18


g_by_random = Graph(13)
# g_by_vertex_order = Graph(13)
by_random(g_by_random)
# by_vertex_order(g_by_vertex_order)
g_by_random.KruskalMST()
# g_by_vertex_order.KruskalMST()
# compare_contents(g_by_random.result, g_by_vertex_order.result)
print(g_by_random.parent)
# print(g_by_vertex_order.parent)

"""
 There is a difference in how the vertexes are specified.
 This results in different root vertices when building up the MST
"""

"""
    After the weights of all edges have been calculated, the edges will be sorted in ascending order.
    Parent array is initialized which will be the size of all vertices:
        i.e. V = 13 => len(parent) = 13
    The parent array is further populated with the vertices based on their corresponding number
        i.e. parent = [index_0 = vertex_0,
                        index_1 = vertex_1,
                        index_2 = vertex_2,
                        ...
                        index_12 = vertex_12]
    Then we initialize a rank array which will be the size of all vertices:
        i.e. V = 13 => len(rank) = 13
    The rank array is initialized with zeros at every index
    
    Obtaining the tree:
        1.  Get the edge with the lowest weight with its (u, v)
        2.  Check the parent for u and v
            2.0 Checking the parents of both vertices
                2.0.1   Recursive look up the parent of v, until a 'root' is reached which satisfies a case
            (Attaching smaller rank trees under the root of higher ranked trees
            2.1 If both vertices have rank=0 (If both are their own parents), u becomes the parent of v (v becomes part of the tree u belongs in)
                The rank of u is also incremented ?How many trees had been taken by u?
            2.2 If u is a higher ranked tree, u becomes the parent of v (v becomes part of the tree u belongs in)
            2.3 If v is a higher ranked tree, v becomes the parent of u (u becomes part of the tree u belongs in)
            2.4 Parent array is updated based on which vertex becomes a parent and which becomes a child of a parent
                i.e. (u=0, v=1, w=2) parent[0, 1, 2,..., 12], 0 becomes parent of 1 thus parent[0, *0*, 2,..., 12] value
                at index 1 is changed to 0 indicating the parent of v=1 is v=0
"""
