`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// BNN Inference Core - MNIST Digit Classifier
//
// Architecture: 784 -> 512 (BNN) -> 256 (BNN) -> 10 (argmax)
// Latency: 512 + 256 + 10 + 2 = 780 cycles @ 100MHz = 7.8us
//
// Compatible with: Xilinx Vivado 2023.2, Zynq-7000
//////////////////////////////////////////////////////////////////////////////////

module bnn_core (
    input  wire         clk,
    input  wire         rst_n,          // Active-low synchronous reset
    input  wire         start,          // Start inference (latched)
    input  wire [783:0] image_in,       // 784-bit binary image
    output reg  [3:0]   digit_out,      // Predicted digit (0-9)
    output reg          valid,          // Result valid pulse
    output reg          busy            // Inference in progress
);

    // -------------------------------------------------------------------------
    // Architecture Parameters
    // -------------------------------------------------------------------------
    localparam integer N_IN    = 784;
    localparam integer N_H1    = 512;
    localparam integer N_H2    = 256;
    localparam integer N_CLASS = 10;
    localparam integer ACC_W   = 25;    // Accumulator width for output layer

    // -------------------------------------------------------------------------
    // FSM States
    // -------------------------------------------------------------------------
    localparam [2:0]
        S_IDLE   = 3'd0,
        S_L1     = 3'd1,
        S_L2     = 3'd2,
        S_OUT    = 3'd3,
        S_ARGMAX = 3'd4,
        S_DONE   = 3'd5;

    reg [2:0] state;
    reg [9:0] idx;
    reg start_latched;

    // -------------------------------------------------------------------------
    // Memory Path (relative to simulation working directory)
    // -------------------------------------------------------------------------
    localparam MEM_DIR = "mem_files/";

    // -------------------------------------------------------------------------
    // Weight & Threshold ROMs
    // -------------------------------------------------------------------------
    (* ram_style = "block" *) reg [N_IN-1:0]  w1      [0:N_H1-1];   // 512 x 784 binary
    (* ram_style = "block" *) reg [N_H1-1:0]  w2      [0:N_H2-1];   // 256 x 512 binary
    (* ram_style = "distributed" *) reg [9:0] thresh1 [0:N_H1-1];    // 512 thresholds
    (* ram_style = "distributed" *) reg [9:0] thresh2 [0:N_H2-1];    // 256 thresholds
    (* ram_style = "distributed" *) reg       invert1 [0:N_H1-1];    // 512 invert flags
    (* ram_style = "distributed" *) reg       invert2 [0:N_H2-1];    // 256 invert flags

    // Output layer: flattened 10x256 weights + 10 biases
    (* ram_style = "block" *) reg signed [15:0] w_out [0:N_CLASS*N_H2-1];
    (* ram_style = "distributed" *) reg signed [15:0] b_out [0:N_CLASS-1];

    // -------------------------------------------------------------------------
    // Memory Initialization
    // -------------------------------------------------------------------------
    initial begin
        $readmemb({MEM_DIR, "weights_l1.mem"},  w1);
        $readmemb({MEM_DIR, "thresh_l1.mem"},   thresh1);
        $readmemb({MEM_DIR, "invert_l1.mem"},   invert1);
        $readmemb({MEM_DIR, "weights_l2.mem"},  w2);
        $readmemb({MEM_DIR, "thresh_l2.mem"},   thresh2);
        $readmemb({MEM_DIR, "invert_l2.mem"},   invert2);
        $readmemh({MEM_DIR, "weights_out.mem"}, w_out);
        $readmemh({MEM_DIR, "bias_out.mem"},    b_out);
    end

    // -------------------------------------------------------------------------
    // Hidden Layer Registers
    // -------------------------------------------------------------------------
    reg [N_H1-1:0] hidden1;
    reg [N_H2-1:0] hidden2;
    reg signed [ACC_W-1:0] out_acc [0:N_CLASS-1];

    // Argmax registers
    reg [3:0] argmax_idx;
    reg signed [ACC_W-1:0] argmax_max;
    integer ai;

    // -------------------------------------------------------------------------
    // Popcount Functions (synthesizable for Xilinx)
    // -------------------------------------------------------------------------
    function [9:0] popcount784;
        input [N_IN-1:0] vec;
        integer k;
        reg [9:0] cnt;
        begin
            cnt = 10'd0;
            for (k = 0; k < N_IN; k = k + 1)
                cnt = cnt + vec[k];
            popcount784 = cnt;
        end
    endfunction

    function [9:0] popcount512;
        input [N_H1-1:0] vec;
        integer k;
        reg [9:0] cnt;
        begin
            cnt = 10'd0;
            for (k = 0; k < N_H1; k = k + 1)
                cnt = cnt + vec[k];
            popcount512 = cnt;
        end
    endfunction

    // -------------------------------------------------------------------------
    // Output Layer MAC Function
    // -------------------------------------------------------------------------
    function automatic signed [ACC_W-1:0] compute_output;
        input [N_H2-1:0]    hidden;
        input [3:0]         neuron;
        input signed [15:0] bias;
        integer j, base;
        reg signed [ACC_W-1:0] acc;
        reg signed [ACC_W-1:0] wext;
        begin
            acc  = {{(ACC_W-16){bias[15]}}, bias};  // Sign-extend bias
            base = neuron * N_H2;

            for (j = 0; j < N_H2; j = j + 1) begin
                wext = {{(ACC_W-16){w_out[base+j][15]}}, w_out[base+j]};
                // hidden[j]=1 means +1, hidden[j]=0 means -1
                if (hidden[j])
                    acc = acc + wext;
                else
                    acc = acc - wext;
            end

            compute_output = acc;
        end
    endfunction

    // -------------------------------------------------------------------------
    // Main FSM
    // -------------------------------------------------------------------------
    always @(posedge clk) begin
        if (!rst_n) begin
            state         <= S_IDLE;
            valid         <= 1'b0;
            busy          <= 1'b0;
            digit_out     <= 4'd0;
            idx           <= 10'd0;
            hidden1       <= {N_H1{1'b0}};
            hidden2       <= {N_H2{1'b0}};
            start_latched <= 1'b0;
        end else begin
            valid <= 1'b0;

            // Latch start pulse
            if (start)
                start_latched <= 1'b1;

            case (state)

                S_IDLE: begin
                    busy <= 1'b0;
                    if (start_latched) begin
                        start_latched <= 1'b0;
                        idx           <= 10'd0;
                        busy          <= 1'b1;
                        state         <= S_L1;
                    end
                end

                // Layer 1: Process 512 neurons (one per cycle)
                S_L1: begin
                    // XNOR popcount: count matching bits
                    // Result XOR with invert flag for negative BatchNorm gamma
                    hidden1[idx] <= (popcount784(~(image_in ^ w1[idx])) >= thresh1[idx]) ^ invert1[idx];

                    if (idx == N_H1-1) begin
                        idx   <= 10'd0;
                        state <= S_L2;
                    end else begin
                        idx <= idx + 10'd1;
                    end
                end

                // Layer 2: Process 256 neurons (one per cycle)
                S_L2: begin
                    hidden2[idx] <= (popcount512(~(hidden1 ^ w2[idx])) >= thresh2[idx]) ^ invert2[idx];

                    if (idx == N_H2-1) begin
                        idx   <= 10'd0;
                        state <= S_OUT;
                    end else begin
                        idx <= idx + 10'd1;
                    end
                end

                // Output Layer: Compute 10 class scores
                S_OUT: begin
                    out_acc[idx[3:0]] <= compute_output(hidden2, idx[3:0], b_out[idx[3:0]]);

                    if (idx == N_CLASS-1) begin
                        state <= S_ARGMAX;
                    end else begin
                        idx <= idx + 10'd1;
                    end
                end

                // Argmax: Find class with highest score
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
                    valid     <= 1'b1;
                    state     <= S_DONE;
                end

                // Done: Return to idle
                S_DONE: begin
                    valid <= 1'b0;
                    busy  <= 1'b0;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;

            endcase
        end
    end

endmodule
