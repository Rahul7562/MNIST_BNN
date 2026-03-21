// =============================================================================
//  bnn_top.v  –  Binary Neural Network Inference  (784 → 64 → 4)
// =============================================================================
//  Input   : 784-bit binarised 28×28 image (pixel > 127 → 1, else 0)
//  Output  : 4-bit digit prediction  (digit 2 → digit_out = 4'b0010)
//
//  Operation (sequential, one neuron per clock):
//    State LAYER1 : compute 64 hidden neurons using XNOR + popcount + thresh
//    State LAYER2 : compute 4  output bits  using XNOR + popcount (thresh=32)
//    State DONE   : assert valid, hold digit_out
//
//  Total latency : 64 + 4 + 2 = 70 clock cycles after start is asserted
//
//  mem_files needed (place next to this file, or update paths below):
//    mem_files/weights_l1.mem   – 64 lines × 784 binary chars
//    mem_files/weights_l2.mem   –  4 lines ×  64 binary chars
//    mem_files/thresh_l1.mem    – 64 lines × 10-bit binary integers
// =============================================================================

`timescale 1ns / 1ps

module bnn_top (
    input  wire         clk,          // system clock
    input  wire         rst_n,        // active-low sync reset
    input  wire         start,        // pulse high 1 cycle to begin inference
    input  wire [783:0] image_in,     // binarised pixels, MSB = pixel[0,0]
    output reg  [3:0]   digit_out,    // 4-bit predicted digit (0–9)
    output reg          valid         // high for 1 cycle when digit_out is ready
);

// ─────────────────────────────────────────────────────────────────────────────
//  Parameters
// ─────────────────────────────────────────────────────────────────────────────
localparam N_INPUT  = 784;
localparam N_HIDDEN = 64;
localparam N_OUT    = 4;
localparam THRESH2  = 7'd32;    // N_HIDDEN/2 – layer-2 threshold (no BN)

// ─────────────────────────────────────────────────────────────────────────────
//  Weight & threshold memories  (initialised from .mem files)
// ─────────────────────────────────────────────────────────────────────────────
// synthesis translate_off
// (For FPGA synthesis, replace with BRAM primitives or IP block for large ROMs)
// synthesis translate_on

reg [N_INPUT-1:0]  w1      [0:N_HIDDEN-1];  // 64 × 784
reg [N_HIDDEN-1:0] w2      [0:N_OUT-1];     //  4 ×  64
reg [9:0]          thresh1 [0:N_HIDDEN-1];  // 64 thresholds (0–784)

initial begin
    $readmemb("mem_files/weights_l1.mem", w1);
    $readmemb("mem_files/weights_l2.mem", w2);
    $readmemb("mem_files/thresh_l1.mem",  thresh1);
end

// ─────────────────────────────────────────────────────────────────────────────
//  Internal registers
// ─────────────────────────────────────────────────────────────────────────────
reg [N_HIDDEN-1:0] hidden;          // binary hidden layer result
reg [6:0]          neuron_idx;      // current neuron being computed

// Popcount accumulators
reg [9:0] pc1;      // 0–784  (10 bits)
reg [6:0] pc2;      // 0– 64  ( 7 bits)

// ─────────────────────────────────────────────────────────────────────────────
//  State machine
// ─────────────────────────────────────────────────────────────────────────────
localparam [1:0]
    S_IDLE   = 2'd0,
    S_LAYER1 = 2'd1,
    S_LAYER2 = 2'd2,
    S_DONE   = 2'd3;

reg [1:0] state;

// ─────────────────────────────────────────────────────────────────────────────
//  Popcount helpers
//  These for-loop implementations synthesise to adder trees in all major tools.
// ─────────────────────────────────────────────────────────────────────────────
function automatic [9:0] popcount784;
    input [N_INPUT-1:0] vec;
    integer k;
    reg [9:0] cnt;
    begin
        cnt = 10'd0;
        for (k = 0; k < N_INPUT; k = k + 1)
            cnt = cnt + {{9{1'b0}}, vec[k]};
        popcount784 = cnt;
    end
endfunction

function automatic [6:0] popcount64;
    input [N_HIDDEN-1:0] vec;
    integer k;
    reg [6:0] cnt;
    begin
        cnt = 7'd0;
        for (k = 0; k < N_HIDDEN; k = k + 1)
            cnt = cnt + {{6{1'b0}}, vec[k]};
        popcount64 = cnt;
    end
endfunction

// ─────────────────────────────────────────────────────────────────────────────
//  Main sequential logic
// ─────────────────────────────────────────────────────────────────────────────
always @(posedge clk) begin
    if (!rst_n) begin
        state      <= S_IDLE;
        hidden     <= {N_HIDDEN{1'b0}};
        digit_out  <= 4'b0000;
        valid      <= 1'b0;
        neuron_idx <= 7'd0;
        pc1        <= 10'd0;
        pc2        <= 7'd0;
    end else begin
        valid <= 1'b0;              // default: deassert valid every cycle

        case (state)
            // ── Wait for start pulse ──────────────────────────────────────────
            S_IDLE: begin
                if (start) begin
                    neuron_idx <= 7'd0;
                    state      <= S_LAYER1;
                end
            end

            // ── Layer 1: process neuron[neuron_idx] ───────────────────────────
            // XNOR(image_in, w1[i]) counts how many weights match the pixels.
            // In {-1,+1} maths: dot = 2·XNOR_count − N_INPUT
            // Fire if XNOR_count > thresh1[i]  (thresh1 has BN folded in)
            S_LAYER1: begin
                pc1 = popcount784(~(image_in ^ w1[neuron_idx]));   // XNOR

                hidden[neuron_idx] <= (pc1 > thresh1[neuron_idx]) ? 1'b1 : 1'b0;

                if (neuron_idx == N_HIDDEN - 1) begin
                    neuron_idx <= 7'd0;
                    state      <= S_LAYER2;
                end else begin
                    neuron_idx <= neuron_idx + 7'd1;
                end
            end

            // ── Layer 2: process output bit[neuron_idx] ───────────────────────
            // 4 neurons → 4 bits of the predicted digit (neuron 0 = MSB)
            // No BN → fixed threshold = N_HIDDEN/2 = 32
            S_LAYER2: begin
                pc2 = popcount64(~(hidden ^ w2[neuron_idx]));       // XNOR

                // neuron 0 → digit_out[3] (MSB), neuron 3 → digit_out[0] (LSB)
                digit_out[N_OUT - 1 - neuron_idx[1:0]] <=
                    (pc2 > THRESH2) ? 1'b1 : 1'b0;

                if (neuron_idx == N_OUT - 1) begin
                    state <= S_DONE;
                end else begin
                    neuron_idx <= neuron_idx + 7'd1;
                end
            end

            // ── Assert valid for one cycle, return to IDLE ────────────────────
            S_DONE: begin
                valid <= 1'b1;
                state <= S_IDLE;
            end

            default: state <= S_IDLE;
        endcase
    end
end

endmodule
