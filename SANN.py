import numpy
import math
import time
def generateRandomNums():
	#l=numpy.random.ranf(24)
	l=numpy.random.uniform(-10,10,24)
	f=open('randnums4.txt','w')
	for i in l:
		f.write(str(i))
		f.write('\n')
	f.close()
	print(l)
	return 
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
		l=self.openRandomNums('randnums4.txt')
		#connections:[Node0,Node1,Node2,Node3,Node4Bias,Node5]
		connection_list=[3,3,3,3,6,3]
		self.learning_rate=2
		self.thrs_arb_high=.5
		self.training_set_input=['000','001','010','011','100','101','110','111']
		self.training_set_output=['001','010','011','100','101','110','111','000'] 
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
		#add output nodes to nodes list
		self.nodes.append(Node(6,0,[0],0))
		self.nodes.append(Node(7,0,[0],0))
		self.nodes.append(Node(8,0,[0],0))
		#change all nodes to correct initial thought
		self.changeInputs()
		bias=1
		self.bias=bias
		self.nodes[4].setValueTo(bias)
		#all weights -3 is excluding the final 3 output nodes because the have no weights
		self.all_weights=[]
		for i in range(len(self.nodes)-3):
			self.all_weights.extend(self.nodes[i].getWeights())
		self.all_weights.extend([l[-3],l[-2],l[-1]])
		#print(self.all_weights[17])
		self.nodes.append(Node(9,3,[l[-3],l[-2],l[-1]],0))
		#self.think()
		self.prompt_questions(self.openRandomNums('final_weights.txt'))
		#for i in self.nodes:
			#i.printInfo()
		return;
	def think(self):
		t1=[5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5]
		t2=[]
		#for i in self.nodes:
			#i.printInfo()
		max_iteration_thoughts=400000
		current_iteration=1
		
		final_error=0.0
		lowest_error=True
		check=[]
		while(lowest_error):
			if current_iteration>max_iteration_thoughts:
				break
			for i in range(len(self.training_set_input)):
				
				#if i==8:
					#self.increase_training_iteration()
					#self.increase_training_iteration()
					#self.increase_training_iteration()
					#self.increase_training_iteration()
					#self.increase_training_iteration()
					#self.increase_training_iteration()
					#break
					
				#print(self.getAllWeights())
				#Hidden Net 1/Sigmoid
				a=self.nodes[0].getWeightedValues()[0]+self.nodes[1].getWeightedValues()[0]+self.nodes[2].getWeightedValues()[0]+self.nodes[4].getWeightedValues()[0]
				self.nodes[3].setValueTo(self.sigmoid(a))
				#Hidden Net 2/Sigmoid
				b=self.nodes[0].getWeightedValues()[1]+self.nodes[1].getWeightedValues()[1]+self.nodes[2].getWeightedValues()[1]+self.nodes[4].getWeightedValues()[1]
				self.nodes[5].setValueTo(self.sigmoid(b))
				#Hidden Net 3/Sigmoid
				c=self.nodes[0].getWeightedValues()[2]+self.nodes[1].getWeightedValues()[2]+self.nodes[2].getWeightedValues()[2]+self.nodes[4].getWeightedValues()[2]
				self.nodes[9].setValueTo(self.sigmoid(c))
				
				#Outputs Net/Sigmoid
				f1=self.nodes[3].getWeightedValues()[0]+self.nodes[4].getWeightedValues()[3]+self.nodes[5].getWeightedValues()[0]+self.nodes[9].getWeightedValues()[0]
				self.nodes[6].setValueTo(self.sigmoid(f1))
				v1=self.nodes[6].getValue()
				f2=self.nodes[3].getWeightedValues()[1]+self.nodes[4].getWeightedValues()[4]+self.nodes[5].getWeightedValues()[1]+self.nodes[9].getWeightedValues()[1]
				self.nodes[7].setValueTo(self.sigmoid(f2))
				v2=self.nodes[7].getValue()
				f3=self.nodes[3].getWeightedValues()[2]+self.nodes[4].getWeightedValues()[5]+self.nodes[5].getWeightedValues()[2]+self.nodes[9].getWeightedValues()[2]
				self.nodes[8].setValueTo(self.sigmoid(f3))
				v3=self.nodes[8].getValue()
				
				#Partial Errors
				#self.thrs_arb_high=.5
				tempO=self.training_set_output[self.train_iteration]
				e1=self.squared_error(v1,float(tempO[0]))
				e2=self.squared_error(v2,float(tempO[1]))
				e3=self.squared_error(v3,float(tempO[2]))
				etotal=e1+e2+e3
				final_error=etotal
				#if final_error<=.000005:
					#print('lowest error')
					#lowest_error=False
					#break
				
				
				#if self.training_set_input[self.train_iteration]=='000':
				check.append(self.printBinaryGuess(v1,v2,v3,self.thrs_arb_high))
				s1=self.tester(self.training_set_input[self.train_iteration],t1,v1,0)
				s2=self.tester(self.training_set_input[self.train_iteration],t1,v2,1)
				s3=self.tester(self.training_set_input[self.train_iteration],t1,v3,2)
				t2.append(v1)
				t2.append(v2)
				t2.append(v3)
				print('output 1: '+str(v1) +' '+s1)
				print('output 2: '+str(v2) +' '+s2)
				print('output 3: '+str(v3) +' '+s3)
				
				part_eo1=self.pderiv_Errortotal_output(v1,float(tempO[0]))
				part_eo2=self.pderiv_Errortotal_output(v2,float(tempO[1]))
				part_eo3=self.pderiv_Errortotal_output(v3,float(tempO[2]))
				part_outn1=self.pderiv_Sig_net(v1)
				part_outn2=self.pderiv_Sig_net(v2)
				part_outn3=self.pderiv_Sig_net(v3)
				
				#Adjust Weights(Backprop)
				temp_al=self.copied_List(self.getAllWeights()) 
				ws=self.getAllWeights() 
				n=self.nodes 
				r=self.learning_rate
				
				#Output 1 weights 
				temp_al[9]=ws[9]-r*(part_eo1*part_outn1*self.sigmoid(a))
				temp_al[18]=ws[18]-r*(part_eo1*part_outn1*self.sigmoid(b))
				temp_al[15]=ws[15]-r*(part_eo1*part_outn1*n[4].getValue())
				temp_al[21]=ws[21]-r*(part_eo1*part_outn1*self.sigmoid(c))
				#Output 2 weights
				temp_al[10]=ws[10]-r*(part_eo2*part_outn2*self.sigmoid(a))
				temp_al[19]=ws[19]-r*(part_eo2*part_outn2*self.sigmoid(b))
				temp_al[16]=ws[16]-r*(part_eo2*part_outn2*n[4].getValue())
				temp_al[22]=ws[22]-r*(part_eo2*part_outn2*self.sigmoid(c))
				#Output 3 weights
				temp_al[11]=ws[11]-r*(part_eo3*part_outn3*self.sigmoid(a))
				temp_al[20]=ws[20]-r*(part_eo3*part_outn3*self.sigmoid(b))
				temp_al[17]=ws[17]-r*(part_eo3*part_outn3*n[4].getValue())
				temp_al[23]=ws[23]-r*(part_eo3*part_outn3*self.sigmoid(c))
				#Hidden Layer 1 weights 
				part_eo1_outh1=part_eo1*part_outn1*ws[9]
				part_eo2_outh1=part_eo2*part_outn2*ws[10]
				part_eo3_outh1=part_eo3*part_outn3*ws[11]
				part_etotal_outh1=part_eo1_outh1+part_eo2_outh1+part_eo3_outh1
				part_outh1_neth1=self.pderiv_Sig_net(self.sigmoid(a))
				temp_al[0]=ws[0]-r*part_etotal_outh1*part_outh1_neth1*n[0].getValue()
				temp_al[3]=ws[3]-r*part_etotal_outh1*part_outh1_neth1*n[1].getValue()
				temp_al[6]=ws[6]-r*part_etotal_outh1*part_outh1_neth1*n[2].getValue()
				temp_al[12]=ws[12]-r*part_etotal_outh1*part_outh1_neth1*n[4].getValue()
				#Hidden Layer 2 weights
				part_eo1_outh2=part_eo1*part_outn1*ws[18]
				part_eo2_outh2=part_eo2*part_outn2*ws[19]
				part_eo3_outh2=part_eo3*part_outn3*ws[20]
				part_etotal_outh2=part_eo1_outh2+part_eo2_outh2+part_eo3_outh2
				part_outh2_neth2=self.pderiv_Sig_net(self.sigmoid(b))
				temp_al[1]=ws[1]-r*part_etotal_outh2*part_outh2_neth2*n[0].getValue()
				temp_al[4]=ws[4]-r*part_etotal_outh2*part_outh2_neth2*n[1].getValue()
				temp_al[7]=ws[7]-r*part_etotal_outh2*part_outh2_neth2*n[2].getValue()
				temp_al[13]=ws[13]-r*part_etotal_outh2*part_outh2_neth2*n[4].getValue()
				#Hidden Layer 3 weights
				part_eo1_outh3=part_eo1*part_outn1*ws[21]
				part_eo2_outh3=part_eo2*part_outn2*ws[22]
				part_eo3_outh3=part_eo3*part_outn3*ws[23]
				part_etotal_outh3=part_eo1_outh3+part_eo2_outh3+part_eo3_outh3
				part_outh3_neth3=self.pderiv_Sig_net(self.sigmoid(c))
				temp_al[2]=ws[2]-r*part_etotal_outh3*part_outh3_neth3*n[0].getValue()
				temp_al[5]=ws[5]-r*part_etotal_outh3*part_outh3_neth3*n[1].getValue()
				temp_al[8]=ws[8]-r*part_etotal_outh3*part_outh3_neth3*n[2].getValue()
				temp_al[14]=ws[14]-r*part_etotal_outh3*part_outh3_neth3*n[4].getValue()
				
				print(temp_al)
				self.updateAllWeights(temp_al)
				self.increase_training_iteration()
				self.changeInputs()
				#break
				
			
			t1=self.replace(t1,t2)
			t2=[]
			print('current iteration: '+str(current_iteration))
			print('===============================================================================')
			if check[0]=='|I believe the number after 0 should be 1|'and check[1]=='|I believe the number after 1 should be 2|' and check[2]=='|I believe the number after 2 should be 3|'and check[3]=='|I believe the number after 3 should be 4|'and check[4]=='|I believe the number after 4 should be 5|'and check[5]=='|I believe the number after 5 should be 6|'and check[6]=='|I believe the number after 6 should be 7|'and check[7]=='|I believe the number after 7 should be 0|':
				print('conditions met!')
				lowest_error=False
				break
			check=[]
			current_iteration+=1
			
		
		self.write_weights(final_error,current_iteration)	
		self.prompt_questions(self.getAllWeights())
		return
		
		
		
	def binaryOperation(self,output,input):
		#t - change
		#f - dont change 
		l=[]
		if input==7:
			n=input-7
			c=bin(n)
			d=self.pythonBinToBinary(str(c),n)
			e=self.pythonBinToBinary(str(bin(output)),output)
			for i in range(len(d)):
				if d[i]==e[i]:
					l.append('f')
					continue
				l.append('t')
			#print(l)	
			return l
		n=input+1
		c=bin(n)
		d=self.pythonBinToBinary(str(c),n)
		e=self.pythonBinToBinary(str(bin(output)),output)
		for i in range(len(d)):
			if d[i]==e[i]:
				l.append('f')
				continue
			l.append('t')
		#print(l)	
		return l
	def pythonBinToBinary(self,s,n):
		if n==1 or n==0:
			ns=s[2:]
			ns='00'+ns
			return ns
		if n==3 or n==2:
			ns=s[2:]
			ns='0'+ns
			return ns
		ns=s[2:]
		return ns
	def percentToBinary(self,f1,f2,f3,high):
		sa=''
		sb=''
		sc=''
		if f1>=high:
			sa='1'
		else:
			sa='0'
		if f2>=high:
			sb='1'
		else:
			sb='0'
		if f3>=high:
			sc='1'
		else:
			sc='0'
		return sa+sb+sc
	def printBinaryGuess(self,f1,f2,f3,high):
		s1=self.training_set_input[self.train_iteration]
		sa=''
		sb=''
		sc=''
		if f1>high:
			sa='1'
		else:
			sa='0'
		if f2>high:
			sb='1'
		else:
			sb='0'
		if f3>high:
			sc='1'
		else:
			sc='0'
		s2=sa+sb+sc
		finalstr='|I believe the number after '+self.binaryToInt(s1)+ ' should be '+ self.binaryToInt(s2)+'|'
		print('------------------------------------------')
		print(finalstr)
		print('------------------------------------------')
		return finalstr
	def binaryToInt(self,s):
		n=0
		if s=='000':
			n='0'
		elif s=='001':
			n='1'
		elif s=='010':
			n='2'
		elif s=='011':
			n='3'
		elif s=='100':
			n='4'
		elif s=='101':
			n='5'
		elif s=='110':
			n='6'
		elif s=='111':
			n='7'
		return n
	def getAllWeights(self):
		return self.all_weights
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
		l2=[self.all_weights[-3],self.all_weights[-2],self.all_weights[-1]]
		self.nodes[9].updateWeights(l2)
		return self.all_weights
	def changeInputs(self):
		for i in range(3):
			self.nodes[i].setValueTo(int(self.training_set_input[self.train_iteration][i]))
	def increase_training_iteration(self):
		self.train_iteration+=1
		if self.train_iteration==len(self.training_set_input):
			self.train_iteration=0
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
	def sigmoid(self,x):
		y=1/(1+math.pow(math.e,-x))
		return y
	def squared_error(self,prediction,target):
		return .5*(math.pow(prediction-target,2))
	def pderiv_Errortotal_output(self,prediction,target):
		return (prediction-target)
	def pderiv_Sig_net(self,o):
		return o*(1-o)
	def copied_List(self,l):
		l2=[]
		for i in l:
			l2.append(i)
		return l2
	def write_weights(self,t,m):
		f=open('best_found_weights.txt','w')
		l=self.getAllWeights()
		for i in range(len(l)):
			f.write('w'+str(i)+': '+str(l[i])+'\n')
		f.write('Total error for weights: '+str(t)+'\n')
		f.write('Max iterations: '+str(m))
		f.close()
		return 
	def tester(self,i,l,value,r):
		c=0
		if i=='000':
			c=0
		elif i=='001':
			c=3
		elif i=='010':
			c=6
		elif i=='011':
			c=9
		elif i=='100':
			c=12
		elif i=='101':
			c=15
		elif i=='110':
			c=18
		elif i=='111':
			c=21
			
		if l[c+r]<value:
			return 'increasing'
		
		
		return 'decreasing'
	def replace(self,l1,l2):
		s=l1
		for i in range(len(l2)):
			s[i]=l2[i]
		return s
	def prompt_questions(self,l):
		#print(l)
		greet=input("ANN: I can count from 0-7 in binary. Ask me a question! ")
		print("You: "+greet)
		print("ANN: Hmmm... let me think.")
		n=self.prompt_support(greet)
		if n==-1:
			print('ANN: alright. I gotta head out')
			return
		another_question=True
		while(another_question==True):
			bin=self.training_set_input[n]
			neth1=float(bin[0])*l[0]+float(bin[1])*l[3]+float(bin[2])*l[6]+self.bias*l[12]
			neth2=float(bin[0])*l[1]+float(bin[1])*l[4]+float(bin[2])*l[7]+self.bias*l[13]
			neth3=float(bin[0])*l[2]+float(bin[1])*l[5]+float(bin[2])*l[8]+self.bias*l[14]
			outh1=self.sigmoid(neth1)
			outh2=self.sigmoid(neth2)
			outh3=self.sigmoid(neth3)
			neto1=outh1*l[9]+outh2*l[18]+outh3*l[21]+self.bias*l[15]
			neto2=outh1*l[10]+outh2*l[19]+outh3*l[22]+self.bias*l[16]
			neto3=outh1*l[11]+outh2*l[20]+outh3*l[23]+self.bias*l[17]
			v1=self.sigmoid(neto1)
			v2=self.sigmoid(neto2)
			v3=self.sigmoid(neto3)
			print(v1)
			print(v2)
			print(v3)
			
			a=self.percentToBinary(v1,v2,v3,self.thrs_arb_high)
			index=0
			for i in range(len(self.training_set_input)):
				if self.training_set_input[i]==a:
					index=i 
					break
			print("ANN: The number after "+str(n)+" in binary is "+a+ " a.k.a... "+ str(index)+".")
			rep=input("ANN: Any other questions? ")
			print("You: "+rep)
			if 'yes' in rep or 'y' in rep :
				q=input("ANN: Yes? ")
				print("You: "+q)
				print("ANN: Hmmm... let me think.")
				n=self.prompt_support(q)
				continue
			else:
				print("ANN: Goodbye!")
				another_question=False
				break
			
			return 
			
			
		
		
		return 
	def prompt_support(self,s):
		n=0
		if '0' in s:
			n=0
		elif '1' in s:
			n=1
		elif '2' in s:
			n=2
		elif '3' in s:
			n=3
		elif '4' in s:
			n=4
		elif '5' in s:
			n=5
		elif '6' in s:
			n=6
		elif '7' in s:
			n=7
		else:
			return -1
		
		return n

a=ANN()


