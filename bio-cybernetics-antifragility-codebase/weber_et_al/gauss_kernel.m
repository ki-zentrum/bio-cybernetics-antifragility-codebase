% computes the Gaussian kernel given params
function y = gauss_kernel(Norm, n, s, bmu)
y = zeros(n, 1);
for idx = 1:n
    if(bmu~=0)
        y(idx) = Norm * exp(-(idx-bmu)^2/(2*s^2));
    else
        y(idx) = Norm * exp(-(idx-n)^2/(2*s^2));
    end
end
end