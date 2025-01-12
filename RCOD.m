
function Ainv=RCOD(H)

[N,n]=size(H);
[U,R,P]=qr(H,0);
P=sparse(P,1:n,1);
d=abs(diag(R));
tol=max(N,n)*eps(max(d));
r=sum(d>tol);
R2=R(1:r,:)';
[Q,S]=qr(R2,0);
U1=U(:,1:r);
C11=S';
V1=P*Q;
part1=V1/C11;
Ainv=part1*U1';

end