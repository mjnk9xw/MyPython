import matplotlib.pyplot as pl
import networkx as nx
import numpy as np

class Program:
    countPoint = x1 = x2 = y1 = y2 = 0
    arrResult = []
    def inputAll(self):
        print('nhập số lượng điểm: ')
        self.countPoint = int(input())
        print('nhập lần lượt 2 điểm bất kì để chọn đường đi ngắn nhất: ')
        print('tọa độ điểm thứ nhất: ')
        self.x1 = int(input())
        self.y1 = int(input())
        print('tọa độ điểm thứ hai: ')
        self.x2 = int(input())
        self.y2 = int(input())
        self.rand()

    def viewNetworkx(self,all):
        import networkx as nx
        import numpy as np
        import matplotlib.pyplot as plt
        import pylab
        
        G = nx.DiGraph(directed=True)
        val_map = {}
        red_edges = []
        if all == True:
            val_map[0] = 1.0
            for i in range(self.countPoint):
                for j in range(self.countPoint):
                    if self.graph[i,j] > 0 and self.listWay[i][j] == 1:
                        G.add_edges_from([(i, j)], weight=int(self.graph[i,j]))

            for i in range(len(self.arrResult)-1):
                    u = self.arrResult[i]
                    v = self.arrResult[i+1]
                    print(u,v)
                    if self.graph[u,v] > 0 and self.listWay[u,v] == 1:
                        val_map[u] = 1.0
                        red_edges.append((u,v))
            val_map[self.arrResult[len(self.arrResult)-1]] = 1.0
            
        else:
            for i in range(self.countPoint):
                for j in range(self.countPoint):
                    if self.graph[i,j] > 0 and self.listWay[i,j] == 1:
                        G.add_edges_from([(i, j)], weight=int(self.graph[i,j]))
        
        print(red_edges)
        print(val_map)

        values = [val_map.get(node, 0.45) for node in G.nodes()]
        edge_labels=dict([((u,v,),d['weight'])
                        for u,v,d in sorted(G.edges(data=True),key=lambda x: x[2]['weight'])])

        edge_colors = ['grey' if not edge in red_edges else 'red' for edge in G.edges()]
        
        pos=nx.spring_layout(G)
        nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels)
        nx.draw(G,pos, node_color = values, node_size=300,edge_color=edge_colors,edge_cmap=plt.cm.Reds,with_labels=True)
        pylab.show()

    # def viewPlot(self):
    #     pl.scatter(self.randData[:,0],self.randData[:,1],50,'blue','s')
      
    #     if len(self.arrResult) > 0:
    #         pl.plot(self.randData[self.arrResult[:],0],self.randData[self.arrResult[:],1],'go-')

    #     pl.scatter(self.x1,self.y1, c='r')
    #     pl.scatter(self.x2,self.y2, c='r')
    #     pl.show()

    def rand(self):
        self.randData = np.random.randint(-100,100,(self.countPoint,2)) 
        begin = np.array((self.x1,self.y1)) 
        end = np.array((self.x2,self.y2))
        self.countPoint += 1
        self.randData = np.insert(self.randData,[0],begin).reshape(self.countPoint, 2)
        self.countPoint += 1
        self.randData = np.insert(self.randData,len(self.randData)*2,end).reshape(self.countPoint, 2)
        print(self.randData)
        
        # 0 -> không tồn tại đường đi
        # 1 -> tồn tại đượng đi từ i->j
        # self.listWay = np.random.randint(-1,3,(self.countPoint,self.countPoint))
        self.listWay = np.ones((self.countPoint,self.countPoint),dtype=np.int)
        for i in range(0,self.countPoint,1):
            for j in range(0,self.countPoint,1):
                # điểm đầu vào cuối không có đường đi.
                randWay = 0
                if i == j:
                    randWay = 0
                elif i == 0 and j == self.countPoint - 1:
                    #không tồn tại trực tiếp đường đi từ điểm bắt đầu -> điểm kết thúc ( nếu không thì đi sẽ rất nhanh )                    
                    randWay = 0
                elif i != 0 and j == self.countPoint - 1:
                    #tồn tại tất cả các đường đi từ điểm ( i = 1 -> n - 2 ) -> điểm kết thúc
                    randWay = 1
                else:
                    randWay = np.random.randint(0,2)
                self.listWay[i][j] = randWay
        print(self.listWay)
        self.makeGraphDistance()

    def process(self):
        print('nhấn `ok` để tìm kiếm đường đi 2 điểm đã chọn: ')
        if input() == '':
            self.dijkstra(0,self.countPoint-1)

    def minDistance(self, dist,u, checkVisit): 
        min = 100000 
        min_index = -1
        for v in range(self.countPoint): 
            if dist[v] < min and checkVisit[v] == False and self.listWay[u][v] == 1: 
                min = dist[v] 
                min_index = v 
        return min_index 
    
    def makeGraphDistance(self):
        self.graph = np.zeros((self.countPoint,self.countPoint))
        for i in range(self.countPoint):
            for j in range(self.countPoint):
                if self.listWay[i][j] == 0:
                    self.graph[i][j] = -1
                else:
                    begin = self.randData[i]
                    end = self.randData[j]
                    self.graph[i][j] = int(np.sqrt(np.power(end[0] - begin[0],2) + np.power(end[1] - begin[1],2)))
        print(self.graph)

    def dijkstra(self,src, endpoint):
        print('processing ...............')
        dist = [100000] * self.countPoint
        dist[src] = 0
        parent = [0] * self.countPoint
        parent[0] = -1
        checkVisit = [False] * self.countPoint 
   
        for cout in range(self.countPoint): 
            u = self.minDistance(dist,cout, checkVisit) 
            # print(u)
            if u >= 0:
                checkVisit[u] = True
                # self.arrResult.append(u)
                # if u == endpoint:
                #     print(self.arrResult)
                # return 
                for v in range(self.countPoint): 
                    if self.graph[u][v] > 0 and checkVisit[v] == False and dist[v] > dist[u] + self.graph[u][v] and self.listWay[u][v] == 1: 
                        dist[v] = dist[u] + self.graph[u][v]
                        parent[v] = u

        self.printSolution(dist,parent)

    def printPath(self, parent, j, appendOk): 
        if parent[j] == -1 :  
            print(j)
            return
        self.printPath(parent , parent[j],appendOk) 
        print(j)
        if appendOk == True:
            self.arrResult.append(j)
        
    def printSolution(self, dist, parent): 
        src = 0
        self.arrResult.append(src)
        print("Vertex \t\tDistance from Source\tPath") 
        for i in range(1, len(dist)): 
            print("\n%d --> %d \t\t%d \t\t" % (src, i, dist[i])), 
            if i == self.countPoint - 1:
                self.printPath(parent,i, True)
            else:
                self.printPath(parent,i, False)
        
        if len(self.arrResult) == 2 and dist[self.arrResult[1]] == 100000:
            print("Không thể tìm được đường đi giữa điểm 2 điểm: 0 và ",self.arrResult[1])
        else:
            print("Đường đi giữ 2 điểm: ",self.arrResult.__str__())
            print("Khoảng cách: ",dist[self.countPoint-1])

main = Program()
main.inputAll()
main.viewNetworkx(False)
main.process()
main.viewNetworkx(True)