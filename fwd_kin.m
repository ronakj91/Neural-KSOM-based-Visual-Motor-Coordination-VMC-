function X = fwd_kin(theta)
l1 = 0.254;
l2 = 0.254;
l3 = 0.254;
t = 0.05;
R = l2*cos(theta(2))+l3*cos(theta(3))+t;
x = R*cos(theta(1));
y = R*sin(theta(1));
z = l2*sin(theta(2))+l3*sin(theta(3))+l1;
X = [x;y;z];
end