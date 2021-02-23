import numpy
import math
class Node:
	def __init__(self,tag,connections,weights,value):
		self.tag=tag
		self.connections=connections
		self.value=value
		self.weights=weights 
		return
	def printInfo(self):
		print('Node: ' +str(self.tag))
		print('Connections: '+str(self.connections))
		print('Value: ' +str(self.value))
		for i in range(len(self.weights)):
			s='w'+str(i+1)+': '+str(self.weights[i])
			print(s)
		l=self.getWeightedValues()
		for i in range(len(l)):
			s='weighted_value'+str(i+1)+': '+str(l[i])
			print(s)
		print('===================')
		return	
	def getValue(self):
		return self.value
	def getWeights(self):
		return self.weights
	def getWeightedValues(self):
		l=[]
		for i in self.weights:
			l.append(i*self.value)
		return l
	def getNumOfWeights(self):
		return len(self.getWeights())
	def updateWeights(self,l):
		self.weights=l
		return self.weights
		
	def setValueTo(self,n):
		self.value=n
class ANN:
	def __init__(self):
		l=self.openRandomNums('randnums2.txt')
		#connections:[i1,i2,b1,h1,h2,b2]
		connection_list=[2,2,2,2,2,2]
		self.learning_rate=.5
		self.training_set_input=[(.05,.10)]
		self.training_set_output=[(.01,.99)] 
		self.train_iteration=0
		nodeinfo=[]
		self.nodes=[]
		tag=0
		c=0
		index=0
		#break node information into list of tuples
		for i in range(len(l)):
			t=()
			templ=[]
			if c==len(connection_list):
				break
			for j in range(connection_list[c]):
				templ.append(l[index])
				index+=1
			t=t+(tag,connection_list[c],templ,0)
			c+=1	
			tag+=1
			nodeinfo.append(t)
		#assign the individual information a specific node and add to node list Nodes(0-5)
		for i in nodeinfo:
			tag=0
			connect=0
			templ=[]
			v=0
			for j in range(len(i)):
				if j==0:
					tag=i[j]
				elif j==1:
					connect=i[j]
				elif j==2:
					templ=i[j]
				elif j==3:
					v=i[j]
			n=Node(tag,connect,templ,v)
			self.nodes.append(n)
		
		#change all nodes to correct initial thought
		#self.changeInputs()
		bias=1
		self.nodes.append(Node(6,0,[0],0))
		self.nodes.append(Node(7,0,[0],0))
		self.nodes[0].setValueTo(.05)
		self.nodes[1].setValueTo(.1)
		self.nodes[2].setValueTo(1)
		self.nodes[5].setValueTo(1)
		
		self.all_weights=[]
		for i in range(len(self.nodes)-2):
			self.all_weights.extend(self.nodes[i].getWeights())
		
		for i in range(10000):
			self.think()
		return;	
		
	def think(self):
		a=self.nodes[0].getWeightedValues()[0] + self.nodes[1].getWeightedValues()[0] + self.nodes[2].getWeightedValues()[0]
		self.nodes[3].setValueTo(self.sigmoid(a))
		b=self.nodes[0].getWeightedValues()[1] + self.nodes[1].getWeightedValues()[1] + self.nodes[2].getWeightedValues()[1]
		self.nodes[4].setValueTo(self.sigmoid(b))
		
		f1=self.nodes[3].getWeightedValues()[0] + self.nodes[4].getWeightedValues()[0] + self.nodes[5].getWeightedValues()[0]
		self.nodes[6].setValueTo(self.sigmoid(f1))
		v1=self.nodes[6].getValue()
		f2=self.nodes[3].getWeightedValues()[1] + self.nodes[4].getWeightedValues()[1] + self.nodes[5].getWeightedValues()[1]
		self.nodes[7].setValueTo(self.sigmoid(f2))
		v2=self.nodes[7].getValue()
		print('output 1: '+str(v1))
		print('output 2: '+str(v2))
		
		tempO=self.training_set_output[self.train_iteration]
		e1=self.squared_error(v1,float(tempO[0]))
		e2=self.squared_error(v2,float(tempO[1]))
		etotal=e1+e2
		part_eo1=self.pderiv_Errortotal_output(v1,float(tempO[0]))
		part_eo2=self.pderiv_Errortotal_output(v2,float(tempO[1]))
		part_outn1=self.pderiv_Sig_net(v1)
		part_outn2=self.pderiv_Sig_net(v2)
		print('total error: '+str(etotal))
		temp_al=self.copied_List(self.getAllWeights())
		ws=self.getAllWeights()
		n=self.nodes
		r=self.learning_rate
		#print(self.getAllWeights())
		#Output 1 Weights *1. Do not alter bias weights
		temp_al[6]=ws[6]-r*(part_eo1*part_outn1*self.sigmoid(a))
		temp_al[8]=ws[8]-r*(part_eo1*part_outn1*self.sigmoid(b))
		#Output 2 Weights
		temp_al[7]=ws[7]-r*(part_eo2*part_outn2*self.sigmoid(a))
		temp_al[9]=ws[9]-r*(part_eo2*part_outn2*self.sigmoid(b))
		#Hidden Layer 1 weights 
		part_eo1_outh1=part_eo1*part_outn1*ws[6]
		part_eo2_outh1=part_eo2*part_outn2*ws[7]
		part_etotal_outh1=part_eo1_outh1+part_eo2_outh1
		part_outh1_neth1=self.pderiv_Sig_net(self.sigmoid(a))
		temp_al[0]=ws[0]-r*part_etotal_outh1*part_outh1_neth1*n[0].getValue()
		temp_al[2]=ws[2]-r*part_etotal_outh1*part_outh1_neth1*n[1].getValue()
		#Hidden Layer 2 weights 
		part_eo1_outh2=part_eo1*part_outn1*ws[8]
		part_eo2_outh2=part_eo2*part_outn2*ws[9]
		part_etotal_outh2=part_eo1_outh2+part_eo2_outh2
		part_outh1_neth2=self.pderiv_Sig_net(self.sigmoid(b))
		temp_al[1]=ws[1]-r*part_etotal_outh2*part_outh1_neth2*n[0].getValue()
		temp_al[3]=ws[3]-r*part_etotal_outh2*part_outh1_neth2*n[1].getValue()
		
		self.updateAllWeights(temp_al)
		#print(self.getAllWeights())
		print('================================')
		
		#for i in self.nodes:
			#i.printInfo()
		
		
		
	def updateAllWeights(self,l):
		#should always be 17 total weights everytime
		self.all_weights=l
		index=0
		for i in range(len(self.nodes)-3):
			tl=[]
			for j in range(self.nodes[i].getNumOfWeights()):
				tl.append(l[index])
				index+=1
			self.nodes[i].updateWeights(tl)	
		return self.all_weights	
	def pderiv_Sig_net(self,o):
		return o*(1-o)	
	def pderiv_Errortotal_output(self,prediction,target):
		return (prediction-target)	
	def getAllWeights(self):
		return self.all_weights	
	def squared_error(self,prediction,target):
		return .5*(math.pow(prediction-target,2))	
		
	def sigmoid(self,x):
		y=1/(1+math.pow(math.e,-x))
		return y	
	
	def copied_List(self,l):
		l2=[]
		for i in l:
			l2.append(i)
		return l2
	
	def openRandomNums(self,file_name):
		try:
			f=open(file_name)
		except:
			return
		l=f.readlines()
		l2=[]
		for i in l:
			l2.append(float(i.strip('\n')))
		return l2;	
a=ANN()		
		
		
		
		
		
		