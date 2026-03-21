`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 20.03.2026 17:02:58
// Design Name: 
// Module Name: top
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


module top (
    input  wire         clk,
    input  wire         rst_n,      // active-low synchronous reset
    input  wire         start,      // can be 1-cycle pulse; latched internally
    input  wire [783:0] image_in,
    output reg  [3:0]   digit_out,
    output reg          valid
);

    // -------------------------------------------------------------------------
    // Constants
    // -------------------------------------------------------------------------
    localparam integer N_IN    = 784;
    localparam integer N_H1    = 512;
    localparam integer N_H2    = 256;
    localparam integer N_CLASS = 10;

    // accumulator width: bias(16) + sum of up to 256 weights(16) => ~25 bits safe
    localparam integer ACC_W   = 25;

    // FSM states
    localparam [2:0]
        S_IDLE   = 3'd0,
        S_L1     = 3'd1,
        S_L2     = 3'd2,
        S_OUT    = 3'd3,
        S_ARGMAX = 3'd4,
        S_DONE   = 3'd5;

    reg [2:0] state;

    // Start latch so we cannot miss a narrow pulse
    reg start_latched;

    // Indices
    reg [9:0] idx; // used for L1 (0..511), L2 (0..255), OUT (0..9)

    // -------------------------------------------------------------------------
    // Memory path
    // Change this if xsim can't find mem_files/ (e.g., set to "" to read from cwd)
    // -------------------------------------------------------------------------
    localparam [8*10-1:0] MEM_DIR = "mem_files/"; // "mem_files/" or ""

    // -------------------------------------------------------------------------
    // Weight & threshold ROMs
    // -------------------------------------------------------------------------
    reg [N_IN-1:0] w1      [0:N_H1-1];  // 512 x 784 (binary)
    reg [N_H1-1:0] w2      [0:N_H2-1];  // 256 x 512 (binary)
    reg [9:0]      thresh1 [0:N_H1-1];
    reg [9:0]      thresh2 [0:N_H2-1];

    // Invert flags for handling negative BatchNorm gamma
    reg            invert1 [0:N_H1-1];
    reg            invert2 [0:N_H2-1];

    // Output layer weights/bias
    // Flattened: w_out[cls*256 + j]
    reg signed [15:0] w_out [0:N_CLASS*N_H2-1]; // 2560 entries
    reg signed [15:0] b_out [0:N_CLASS-1];      // 10 entries

    initial begin
        // NOTE: if MEM_DIR is "mem_files/", Vivado must make that folder visible in xsim run dir.
        $readmemb({MEM_DIR,"weights_l1.mem"},  w1);
        $readmemb({MEM_DIR,"thresh_l1.mem"},   thresh1);
        $readmemb({MEM_DIR,"invert_l1.mem"},   invert1);
        $readmemb({MEM_DIR,"weights_l2.mem"},  w2);
        $readmemb({MEM_DIR,"thresh_l2.mem"},   thresh2);
        $readmemb({MEM_DIR,"invert_l2.mem"},   invert2);

        $readmemh({MEM_DIR,"weights_out.mem"}, w_out);
        $readmemh({MEM_DIR,"bias_out.mem"},    b_out);
    end

    // -------------------------------------------------------------------------
    // Data regs
    // -------------------------------------------------------------------------
    reg [N_H1-1:0] hidden1;
    reg [N_H2-1:0] hidden2;

    reg signed [ACC_W-1:0] out_acc [0:N_CLASS-1];

    // Argmax work regs
    reg [3:0]              argmax_idx;
    reg signed [ACC_W-1:0] argmax_max;
    integer                ai;

    // -------------------------------------------------------------------------
    // Popcount
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

    // masked sum for output layer (binary hidden2)
    function automatic signed [ACC_W-1:0] masked_sum;
    input [255:0]       hidden;     // 1-bit encoding: 1 => +1, 0 => -1
    input [3:0]         neuron;      // class 0..9
    input signed [15:0] bias;        // Q8.8 scaled by 256
    integer j, base;
    reg signed [ACC_W-1:0] acc;
    reg signed [ACC_W-1:0] wext;
    begin
        acc  = {{(ACC_W-16){bias[15]}}, bias};
        base = neuron * 256;

        for (j = 0; j < 256; j = j + 1) begin
            wext = {{(ACC_W-16){w_out[base+j][15]}}, w_out[base+j]};

            // hidden bit encodes +/-1
            if (hidden[j])
                acc = acc + wext;   // +1 * w
            else
                acc = acc - wext;   // -1 * w
        end

        masked_sum = acc;
    end
endfunction

    // -------------------------------------------------------------------------
    // Main sequential logic
    // -------------------------------------------------------------------------
    always @(posedge clk) begin
        if (!rst_n) begin
            state         <= S_IDLE;
            valid         <= 1'b0;
            digit_out     <= 4'd0;
            idx           <= 10'd0;
            hidden1       <= {N_H1{1'b0}};
            hidden2       <= {N_H2{1'b0}};
            start_latched <= 1'b0;
        end else begin
            valid <= 1'b0;

            // latch start (can't be missed)
            if (start)
                start_latched <= 1'b1;

            case (state)

                S_IDLE: begin
                    if (start_latched) begin
                        start_latched <= 1'b0;
                        idx           <= 10'd0;
                        state         <= S_L1;
                    end
                end

                // Layer1: 512 cycles
                S_L1: begin
                    // XOR with invert flag handles negative BatchNorm gamma
                    // Use >= for threshold comparison (BNN standard)
                    hidden1[idx] <= (popcount784(~(image_in ^ w1[idx])) >= thresh1[idx]) ^ invert1[idx];

                    if (idx == N_H1-1) begin
                        idx   <= 10'd0;
                        state <= S_L2;
                    end else begin
                        idx <= idx + 10'd1;
                    end
                end

                // Layer2: 256 cycles
                S_L2: begin
                    // XOR with invert flag handles negative BatchNorm gamma
                    // Use >= for threshold comparison (BNN standard)
                    hidden2[idx] <= (popcount512(~(hidden1 ^ w2[idx])) >= thresh2[idx]) ^ invert2[idx];

                    if (idx == N_H2-1) begin
                        idx   <= 10'd0;
                        state <= S_OUT;
                    end else begin
                        idx <= idx + 10'd1;
                    end
                end

                // Output: 10 cycles
                S_OUT: begin
                    out_acc[idx[3:0]] <= masked_sum(hidden2, idx[3:0], b_out[idx[3:0]]);

                    if (idx == N_CLASS-1) begin
                        state <= S_ARGMAX;
                    end else begin
                        idx <= idx + 10'd1;
                    end
                end

                // Argmax: 1 cycle (compute argmax of out_acc[0..9])
                S_ARGMAX: begin
                    // Use blocking assignments for local computation within cycle
                    argmax_max = out_acc[0];
                    argmax_idx = 4'd0;
                    for (ai = 1; ai < N_CLASS; ai = ai + 1) begin
                        if (out_acc[ai] > argmax_max) begin
                            argmax_max = out_acc[ai];
                            argmax_idx = ai[3:0];
                        end
                    end
                    digit_out <= argmax_idx;
                    valid     <= 1'b1;  // Assert valid when result is ready
                    state     <= S_DONE;
                end

                // Done: return to idle (valid was set in S_ARGMAX)
                S_DONE: begin
                    valid <= 1'b0;  // Clear valid after one cycle
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;

            endcase
        end
    end

endmodule
