%load training data
load('train.mat','X')
load('train.mat','y')
%initiliza training data 
yt=y';
Xt=X';
n = size(yt,1);
%initialize data set lable is 0 into -1
for i=1:size(yt,1)
    if yt(i,:)==0
        yt(i,:)=-1;
    end
end 

C=1;
y=yt';
G = (y*yt).*(X*Xt);
% train svm using cvx
cvx_begin
    variables alphaslack(n);
    maximize(ones(n,1)'*alphaslack-1/2*alphaslack'*G*alphaslack);
    subject to 
        alphaslack >= 0;
        alphaslack <= C;
        alphaslack'*yt == 0; 
cvx_end

%calculate w
w=sum(alphaslack*y*X);

%w=(alphaslack.*y)'*X
%calculate based on the formula from the SVM note
%max_wx=max(X(find(y==-1),:)*w');
%min_wx=min(X(find(y==1),:)*w');
%b=(max_wx+min_wx)*(-0.5);

[maxi,I]=max(w*X');
b=1/y(I) - sum(w*X(I));
b=b/norm(b);


%pick random s
%b=1/y - w*x
%b=1/yt*ones(n)-X*w';
%0<alpha<b
%b=1/y(y(find(y~=0)),:)-sum(alphaslack(find(alphaslack~=0)))

%load test data
load('test.mat','X')
load('test.mat','y')
testx=X;
testy=y';
for i=1:size(testy,1)
    if testy(i,:)==0
        testy(i,:)=-1;
    end
end
%calculate accuracy with training module
predic=sign(testx*w'+b);
accuracy =sum(testy==predic)/size(testy,1)
dual_w=w;
dual_b=b;
dual_alphaslack=alphaslack;