clear all
close all
clc;

%% Physical constants
speed_of_light = 3 * 10^8;  % [m/s]

%% Radar Specifications 
%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Frequency of operation = 77 GHz
% Max Range = 200 m
% Range Resolution = 1 m
% Max Velocity = 70 m/s
%%%%%%%%%%%%%%%%%%%%%%%%%%%
fc = 77 * 10^9;           % [Hz]
max_range = 200;          % [m]
range_resolution = 1;     % [m]
max_velocity = 70;        % [m/s]
velocity_resolution = 3;  % [m/s]

%% User Defined Range and Velocity of target
% define the target's initial position and velocity. 
% Note : Velocity remains contant
target_initial_range = 110;     % [m]
target_initial_velocity = -20;  % [m/s]

%% FMCW Waveform Generation

% Design the FMCW waveform by giving the specs of each of its parameters.
% Calculate the Bandwidth (B), Chirp Time (Tchirp) and Slope (slope) of the FMCW
% chirp using the requirements above.

Bandwidth = speed_of_light / (2.0 * range_resolution);  % [Hz]
Tchirp = 5.5 * 2.0 * max_range / speed_of_light;  % [s]
Slope = Bandwidth / Tchirp;

%The number of chirps in one sequence. Its ideal to have 2^ value for the ease of running the FFT
%for Doppler Estimation. 
Nd=128;                   % #of doppler cells OR #of sent periods % number of chirps

%The number of samples on each chirp. 
Nr=1024;                  %for length of time OR # of range cells

% Timestamp for running the displacement scenario for every sample on each
% chirp
t=linspace(0, Nd*Tchirp, Nr*Nd); %total time for samples

%Creating the vectors for Tx, Rx and Mix based on the total samples input.
Tx=zeros(1,length(t)); %transmitted signal
Rx=zeros(1,length(t)); %received signal
Mix = zeros(1,length(t)); %beat signal

%Similar vectors for range_covered and time delay.
r_t=zeros(1,length(t));
td=zeros(1,length(t));


%% Signal generation and Moving Target simulation
% Running the radar scenario over the time. 

for i=1:length(t)         
    % Update the Range of the Target for constant velocity. 
    r_t(i) = target_initial_range + target_initial_velocity * t(i);
    td(i) = 2.0 * (r_t(i) / speed_of_light);  % Round-trip time delay between transmitted and received signal
    
    % Update the transmitted and received signal. 
    Tx(i) = cos(2*pi*(fc*(t(i)        ) + 0.5*Slope*(t(i)        )^2));
    Rx(i) = cos(2*pi*(fc*(t(i) - td(i)) + 0.5*Slope*(t(i) - td(i))^2));
    
    %Now by mixing the Transmit and Receive generate the beat signal
    %This is done by element wise matrix multiplication of Transmit and
    %Receiver Signal
    Mix(i) = Tx(i) * Rx(i);    
end

%% RANGE MEASUREMENT

% Reshape the vector into Nr*Nd array. Nr and Nd here would also define the size of
% Range and Doppler FFT respectively.
Mix_matrix = reshape(Mix, Nr, Nd);

% Run the FFT on the beat signal along the range bins dimension (Nr) and
% normalize.
fft_range = fft(Mix_matrix, Nr, 1) / Nr;

% Take the absolute value of FFT output
fft_range_abs = abs(fft_range);

% Output of FFT is double sided signal, but we are interested in only one side of the spectrum.
% Hence we throw out half of the samples.
fft_range_abs_half = fft_range_abs(1 : Nr / 2, :);

%plotting the range
figure ('Name','Range from First FFT')

% plot FFT output 
plot(fft_range_abs_half(:, 1)); 
axis ([0 200 0 0.5]);

%% RANGE DOPPLER RESPONSE
% The 2D FFT implementation is already provided here. This will run a 2DFFT
% on the mixed signal (beat signal) output and generate a range doppler
% map.You will implement CFAR on the generated RDM


% Range Doppler Map Generation.

% The output of the 2D FFT is an image that has reponse in the range and
% doppler FFT bins. So, it is important to convert the axis from bin sizes
% to range and doppler based on their Max values.

Mix=reshape(Mix,[Nr,Nd]);

% 2D FFT using the FFT size for both dimensions.
sig_fft2 = fft2(Mix,Nr,Nd);

% Taking just one side of signal from Range dimension.
sig_fft2 = sig_fft2(1:Nr/2,1:Nd);
sig_fft2 = fftshift (sig_fft2);
RDM = abs(sig_fft2);
RDM = 10*log10(RDM) ;

%use the surf function to plot the output of 2DFFT and to show axis in both
%dimensions
doppler_axis = linspace(-100,100,Nd);
range_axis = linspace(-200,200,Nr/2)*((Nr/2)/400);
figure,surf(doppler_axis,range_axis,RDM);

%% CFAR implementation

% Slide Window through the complete Range Doppler Map


% Select the number of Training Cells in both the dimensions.
Tr = 10;
Td = 8;

% Select the number of Guard Cells in both dimensions around the Cell under 
% test (CUT) for accurate estimation
Gr = 4;
Gd = 4;

% offset the threshold by SNR value in dB
offset = 10;

% Create a vector to store noise_level for each iteration on training cells
noise_level = 0;
n_training_cells = 0;

% Create matrix to store output
RDM_after_CFAR = zeros(size(RDM));

% Design a loop such that it slides the CUT across range doppler map by
% giving margins at the edges for Training and Guard Cells.
% For every iteration sum the signal level within all the training
% cells. To sum convert the value from logarithmic to linear using db2pow
% function. Average the summed values for all of the training
% cells used. After averaging convert it back to logarithimic using pow2db.
% Further add the offset to it to determine the threshold. Next, compare the
% signal under CUT with this threshold. If the CUT level > threshold assign
% it a value of 1, else equate it to 0.
for i = Tr+Gr+1 : size(RDM, 1) - (Tr+Gr)
    for j = Td+Gd+1 : size(RDM, 2) - (Td+Gd)
        % Add noise level for all training cells (in Watt, not dBm)
        for p = i-(Tr+Gr) : i+(Tr+Gr)
            for q = j-(Td+Gd) : j+(Td+Gd)
                if abs(i - p) > Gr || abs(j - q) > Gd
                    noise_level = noise_level + db2pow(RDM(p, q));
                    n_training_cells = n_training_cells + 1;
                end
            end
        end
        
        % Compute average and convert back to dbM
        noise_level_avg_dbm = pow2db(noise_level / n_training_cells);
        
        % Compute threshold
        threshold = noise_level_avg_dbm + offset;
        
        % Store into output
        if (RDM(i, j) > threshold)
            RDM_after_CFAR(i, j) = 1;
        end        
    end
end



% The process above will generate a thresholded block, which is smaller 
% than the Range Doppler Map as the CUT cannot be located at the edges of
% matrix. Hence,few cells will not be thresholded. To keep the map size same
% set those values to 0. 

% -> This is already done in the previous step, since the RDM_after_CFAR
% matrix was initialized to 0. Since we didn't loop over cells in the edges
% they keep the value 0.

% Display the CFAR output using the Surf function like we did for Range
% Doppler Response output.
figure,surf(doppler_axis, range_axis, RDM_after_CFAR);
colorbar;


 
 