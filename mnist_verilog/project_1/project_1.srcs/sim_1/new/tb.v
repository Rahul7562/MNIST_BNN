`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 20.03.2026 17:06:00
// Design Name: 
// Module Name: tb
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


module tb;

    parameter integer FOCUS_DIGIT = 0;
    parameter integer CLK_HALF    = 5;      // 10ns period
    parameter integer TIMEOUT     = 20000;  // must be > ~800 cycles

    reg          clk;
    reg          rst_n;
    reg          start;
    reg  [783:0] image_in;

    reg  [3:0]   image_id;
    wire [3:0]   digit_out;
    wire         valid;

    top dut (
        .clk       (clk),
        .rst_n     (rst_n),
        .start     (start),
        .image_in  (image_in),
        .digit_out (digit_out),
        .valid     (valid)
    );

    // Keep this consistent with top.v
    localparam [8*10-1:0] MEM_DIR = "mem_files/";

    // Clock
    initial clk = 1'b0;
    always #(CLK_HALF) clk = ~clk;

    // Image loader
    reg [783:0] img_buf [0:0];

    task load_focus_image;
        begin
            case (FOCUS_DIGIT)
                0: $readmemb({MEM_DIR,"test_image_0.mem"}, img_buf);
                1: $readmemb({MEM_DIR,"test_image_1.mem"}, img_buf);
                2: $readmemb({MEM_DIR,"test_image_2.mem"}, img_buf);
                3: $readmemb({MEM_DIR,"test_image_3.mem"}, img_buf);
                4: $readmemb({MEM_DIR,"test_image_4.mem"}, img_buf);
                5: $readmemb({MEM_DIR,"test_image_5.mem"}, img_buf);
                6: $readmemb({MEM_DIR,"test_image_6.mem"}, img_buf);
                7: $readmemb({MEM_DIR,"test_image_7.mem"}, img_buf);
                8: $readmemb({MEM_DIR,"test_image_8.mem"}, img_buf);
                9: $readmemb({MEM_DIR,"test_image_9.mem"}, img_buf);
                default: begin
                    $display("ERROR: FOCUS_DIGIT must be 0..9");
                    $finish;
                end
            endcase
            image_in = img_buf[0];
        end
    endtask

    integer tc;

    initial begin
        // Dump only what you want to see
        $dumpfile("bnn_minimal.vcd");
        $dumpvars(0, image_id);
        $dumpvars(0, digit_out);
        $dumpvars(0, valid);

        // init
        rst_n    = 1'b0;
        start    = 1'b0;
        image_in = 784'b0;
        image_id = FOCUS_DIGIT[3:0];

        load_focus_image();

        // release reset (sync)
        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        // robust start: hold 3 cycles
        start = 1'b1;
        repeat (3) @(posedge clk);
        start = 1'b0;

        // wait for valid
        tc = 0;
        while (!valid && tc < TIMEOUT) begin
            @(posedge clk);
            tc = tc + 1;
        end

        if (!valid) begin
            $display("TIMEOUT waiting for valid");
            $finish;
        end

        // report
        $display("Input  image_id = %04b (%0d)", image_id, image_id);
        $display("Output digit_out= %04b (%0d)", digit_out, digit_out);

        if (digit_out === image_id)
            $display("PASS");
        else
            $display("FAIL");

        repeat (10) @(posedge clk);
        $finish;
    end

endmodule