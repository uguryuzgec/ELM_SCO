

function Hinv=ORT(Hout,lambda)

Hinv=((eye(size(Hout'*Hout))/lambda)+Hout'*Hout)\Hout';

end