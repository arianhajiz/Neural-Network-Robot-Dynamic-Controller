

clear all

disp('please enter a number between (3,5), (the bigger number  the closer curves)')
shift=input('shift=')

hold off

ax=4;
bx=10;

cx=ax+(bx-ax)*rand(4000,1);

ay=2;
by=8;

cy=ay+(by-ay)*rand(4000,1);

% plot(cx,cy,'g*')


k1=zeros(4000,1);
k2=zeros(4000,1);

for i=1:4000
    z1=(cx(i,1)-7)^2+(cy(i,1)-2)^2-9;
    z2=(cx(i,1)-7)^2+(cy(i,1)-2)^2-4;
    if z1<0&z2>0
        k1(i,1)=cx(i,1);
         k2(i,1)=cy(i,1);
    end
end


%xlim([4,14])
% ylim([-2-shift,5])
grid


ax=6;
bx=14;
cx=bx+(ax-bx)*rand(4000,1);

ay=-2;
by=-8;
cy=by+(ay-by)*rand(4000,1);




k11=zeros(4000,1);
k22=zeros(4000,1);

for i=1:4000
    z1=(cx(i,1)-10)^2+(cy(i,1)+2)^2-9;
    z2=(cx(i,1)-10)^2+(cy(i,1)+2)^2-4;
    if z1<0&z2>0
        k11(i,1)=cx(i,1);
         k22(i,1)=cy(i,1)+shift;
    end
end

q = 1;
r = 1;
for i = 1:size(k11,1)
    if k11(i) ~= 0 && k22(i) ~= 0
        k11_f(r,1) = k11(i);
        k22_f(r,1) = k22(i);
        r = r+1;
    end 
end


for i = 1:size(k1,1)
    if k1(i) ~= 0 && k2(i) ~= 0
        k1_f(q,1) = k1(i);
        k2_f(q,1) = k2(i);
        q = q+1;
    end 
end 


plot(k11_f',k22_f','r+')



hold on
plot(k1_f',k2_f','b*')

w(1,:) = [0 0];
train_index = 0.05;

x_train = [k1_f k2_f;k11_f k22_f];
x1 = [k1_f,k2_f];
x2 = [k11_f,k22_f];
j = 1;
for epoch = 1:20
      e = 1;
      tmp = zeros(size(x_train,1),1);
      for t = 1:size(x_train,1)
          i = randi(size(x_train,1),1);
          while (ismember(i,tmp))
              i = randi(size(x_train,1),1);
          end
          tmp(e) = i;
          e = e+1;
          temp1  = ismember(x_train(i,:),x1);
          temp2  = ismember(x_train(i,:),x2);
          if (and(temp1(1),temp1(2))) 
              d(i) = 1;
          elseif (and(temp2(1),temp2(2)))
              d(i) = -1;
          end
          y(j) = sign(w(j,:)*x_train(i,:)');
          w(j+1,:) = w(j,:) + train_index*(d(i) - y(j))*x_train(i);
          j = j+1;


      end 
  end 


my_x = linspace(0,15,10000);
my_y = linspace(0,15,10000);

plot(my_x,w(end,2)*my_y);
 


