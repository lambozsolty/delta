folder = 'C:\Users\lambo\Google Drive\II. Félév\Mesterséges intelligencia\delta\cars';

for i = 1:20
    if i <= 9
        A = imread(strcat(strcat(strcat(folder), '\image_000',num2str(i), '.jpg')));   
    else
        A = imread(strcat(strcat(strcat(folder), '\image_00',num2str(i), '.jpg')));
    end

    image(A)
    pause(1)
end

function w = offline(x, d, f, gradf, stop)
    [~, n] = size(x);
    w = randn(n,size(d,2));
    epoch = 0;
    lr = 0.01;

    while true
         v = x * w;
         y = f(v);
         e = y - d;
         g = x' * (e .* gradf(v)); %
         w = w - lr * g;
         E = sum(e(:).^2);
         if stop(E, epoch), break; end
         epoch = epoch + 1;
    end
end
 
function res = f(x)
    res = 1/(1+exp(-x));
end



