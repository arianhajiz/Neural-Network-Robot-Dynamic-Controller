clear all;
disp('please enter a number between (3,5), (the bigger number  the closer curves)')
shift = input('shift=');


ax = 4; bx = 10;
cx = ax + (bx - ax) * rand(4000,1);
ay = 2; by = 8;
cy = ay + (by - ay) * rand(4000,1);

k1 = zeros(4000,1);
k2 = zeros(4000,1);

for i = 1:4000
    z1 = (cx(i) - 7)^2 + (cy(i) - 2)^2 - 9;
    z2 = (cx(i) - 7)^2 + (cy(i) - 2)^2 - 4;
    if z1 < 0 && z2 > 0
        k1(i) = cx(i);
        k2(i) = cy(i);
    end
end


ax = 6; bx = 14;
cx = bx + (ax - bx) * rand(4000,1);
ay = -2; by = -8;
cy = by + (ay - by) * rand(4000,1);


k11 = zeros(4000,1);
k22 = zeros(4000,1);


for i = 1:4000
    z1 = (cx(i) - 10)^2 + (cy(i) + 2)^2 - 9;
    z2 = (cx(i) - 10)^2 + (cy(i) + 2)^2 - 4;
    if z1 < 0 && z2 > 0
        k11(i) = cx(i);
        k22(i) = cy(i) + shift;
    end
end

% Hazfe sefr az dataset 
k1(k1 == 0) = NaN;
k2(k2 == 0) = NaN;
k1_f = k1(~isnan(k1));
k2_f = k2(~isnan(k2));

k11(k11 == 0) = NaN;
k22(k22 == 0) = NaN;
k11_f = k11(~isnan(k11));
k22_f = k22(~isnan(k22));

% dorost kardane dadeye train be soorate [x y]
x_train = [k1_f k2_f; k11_f k22_f];

% baraye in qesmat be tedade k1_f 1 va be tedade k11_f -1 migzarim
% dar vaqe haman label gozari
d = [ones(size(k1_f)); -1*ones(size(k11_f))];

% w => x,y,bias (dar code ghabli bias dar nazar gerefte nashode bood 
w = zeros(1,3); 

% Perceptron

figure()
hold on;
tmp = 1;
legends = {};
for train_index = 1:10
    for epoch = 1:5
        for i = 1:size(x_train,1)
            input = [1, x_train(i,:)]; % ezafe kardane 1 baraye bias 
            y = sign(w * input');
            w = w + train_index * (d(i) - y) * input;
        end
    end
    
    % rasme khate joda konande
    my_x = linspace(min([k1_f; k11_f]), max([k1_f; k11_f]), 200); % baraye bedast avardane x ha
    % eshtebahe digar in bood ke x ha ra az sefr dar nazar migereftam ama chon
    % dade ha az sefr shoru nemishavand in kar dorost nist 
    my_y = (-w(1) - w(2)*my_x)/w(3);  %  w1*x + w2*y + bias = 0
    
    plot(my_x, my_y);
    % xlim([min([k1_f; k11_f]), max([k1_f; k11_f])]);
    % ylim([min([k2_f; k22_f]), max([k2_f; k22_f])]);
    title('Perceptron Implementation');
    legends{tmp} = sprintf('train index = %.2f', train_index);
    tmp = tmp +1;
end


plot(k1_f, k2_f, 'b*');
plot(k11_f, k22_f, 'r+');

legend(legends{:},'Class 1', 'Class 2');
hold off




