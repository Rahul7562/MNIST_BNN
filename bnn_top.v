`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company:
// Engineer:
//
// Create Date: 20.03.2026 17:02:58
// Design Name:
// Module Name: bnn_top
// Project Name:
// Target Devices:
// Tool Versions:
// Description:
//
// Dependencies:
//
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
//
//////////////////////////////////////////////////////////////////////////////////


module bnn_top (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         start,
    input  wire [783:0] image_in,
    output reg  [3:0]   digit_out,
    output reg          valid
);

localparam N_IN    = 784;
localparam N_H1    = 512;
localparam N_H2    = 256;
localparam N_CLASS = 10;
localparam ACC_W   = 25;

localparam [2:0]
    S_IDLE   = 3'd0,
    S_LAYER1 = 3'd1,
    S_LAYER2 = 3'd2,
    S_OUTPUT = 3'd3,
    S_ARGMAX = 3'd4,
    S_DONE   = 3'd5;

reg [2:0]  state;
reg [9:0]  neuron_idx;

reg [N_IN-1:0]    w1      [0:N_H1-1];
reg [N_H1-1:0]    w2      [0:N_H2-1];
reg [9:0]         thresh1 [0:N_H1-1];
reg [9:0]         thresh2 [0:N_H2-1];

reg signed [15:0] w_out [0:N_CLASS*N_H2-1];
reg signed [15:0] b_out [0:N_CLASS-1];

initial begin
    $readmemb("mem_files/weights_l1.mem",  w1);
    $readmemb("mem_files/weights_l2.mem",  w2);
    $readmemb("mem_files/thresh_l1.mem",   thresh1);
    $readmemb("mem_files/thresh_l2.mem",   thresh2);
    $readmemh("mem_files/weights_out.mem", w_out);
    $readmemh("mem_files/bias_out.mem",    b_out);
end

reg [N_H1-1:0]          hidden1;
reg [N_H2-1:0]          hidden2;
reg signed [ACC_W-1:0]  out_acc [0:N_CLASS-1];

reg [3:0]               argmax_idx;
reg signed [ACC_W-1:0]  argmax_max;
integer                 ai;

function automatic [9:0] popcount784;
    input [783:0] vec;
    integer k;
    reg [9:0] cnt;
    begin
        cnt = 10'd0;
        for (k = 0; k < 784; k = k + 1)
            cnt = cnt + {{9{1'b0}}, vec[k]};
        popcount784 = cnt;
    end
endfunction

function automatic [9:0] popcount512;
    input [511:0] vec;
    integer k;
    reg [9:0] cnt;
    begin
        cnt = 10'd0;
        for (k = 0; k < 512; k = k + 1)
            cnt = cnt + {{9{1'b0}}, vec[k]};
        popcount512 = cnt;
    end
endfunction

function automatic signed [ACC_W-1:0] masked_sum;
    input [255:0]       hidden;
    input [3:0]         neuron;
    input signed [15:0] bias;
    integer j, base;
    reg signed [ACC_W-1:0] acc;
    begin
        acc  = {{(ACC_W-16){bias[15]}}, bias};
        base = neuron * 256;
        for (j = 0; j < 256; j = j + 1) begin
            if (hidden[j])
                acc = acc + {{(ACC_W-16){w_out[base+j][15]}}, w_out[base+j]};
        end
        masked_sum = acc;
    end
endfunction

always @(posedge clk) begin
    if (!rst_n) begin
        state      <= S_IDLE;
        valid      <= 1'b0;
        digit_out  <= 4'd0;
        neuron_idx <= 10'd0;
        hidden1    <= {N_H1{1'b0}};
        hidden2    <= {N_H2{1'b0}};
    end else begin
        valid <= 1'b0;

        case (state)
            S_IDLE: begin
                if (start) begin
                    neuron_idx <= 10'd0;
                    state      <= S_LAYER1;
                end
            end

            S_LAYER1: begin
                hidden1[neuron_idx] <=
                    (popcount784(~(image_in ^ w1[neuron_idx]))
                     > thresh1[neuron_idx]) ? 1'b1 : 1'b0;

                if (neuron_idx == N_H1 - 1) begin
                    neuron_idx <= 10'd0;
                    state      <= S_LAYER2;
                end else begin
                    neuron_idx <= neuron_idx + 10'd1;
                end
            end

            S_LAYER2: begin
                hidden2[neuron_idx] <=
                    (popcount512(~(hidden1 ^ w2[neuron_idx]))
                     > thresh2[neuron_idx]) ? 1'b1 : 1'b0;

                if (neuron_idx == N_H2 - 1) begin
                    neuron_idx <= 10'd0;
                    state      <= S_OUTPUT;
                end else begin
                    neuron_idx <= neuron_idx + 10'd1;
                end
            end

            S_OUTPUT: begin
                out_acc[neuron_idx[3:0]] <=
                    masked_sum(hidden2,
                               neuron_idx[3:0],
                               b_out[neuron_idx[3:0]]);

                if (neuron_idx == N_CLASS - 1) begin
                    state <= S_ARGMAX;
                end else begin
                    neuron_idx <= neuron_idx + 10'd1;
                end
            end

            S_ARGMAX: begin
                argmax_max = out_acc[0];
                argmax_idx = 4'd0;
                for (ai = 1; ai < N_CLASS; ai = ai + 1) begin
                    if (out_acc[ai] > argmax_max) begin
                        argmax_max = out_acc[ai];
                        argmax_idx = ai[3:0];
                    end
                end
                digit_out <= argmax_idx;
                state     <= S_DONE;
            end

            S_DONE: begin
                valid <= 1'b1;
                state <= S_IDLE;
            end

            default: state <= S_IDLE;
        endcase
    end
end

endmodule
