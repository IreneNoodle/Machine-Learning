% train SVM 
clear all
%soft margin primal
load('train.mat','X')
load('train.mat','y')
yt=y'
n = size(X,2);
%initialize data set the 0 label into -1
for i=1:size(yt,1)
    if yt(i,:)==0
        yt(i,:)=-1;
    end
end
y=yt';
%primal 
C=1;
%define primal train module
cvx_begin
    variables w(n) b xi(8500)
    dual variable alphaslack
    minimize( 1/2*w'*w +C*sum(xi)) 
    subject to
        yt.*(X*w + b)-1+xi>= 0:alphaslack; 
        xi>=0;
cvx_end
%load and initialize test data
load('test.mat','X')
load('test.mat','y')
testx=X;
testy=y';
for i=1:size(testy,1)
    if testy(i,:)==0
        testy(i,:)=-1;
    end
end
%calculate accuracy
predic=sign(testx*w+b);
accuracy =sum(testy==predic)/size(testy,1)
%save the w and b
prim_w=w;
prim_b=b;

plot(prim_w)