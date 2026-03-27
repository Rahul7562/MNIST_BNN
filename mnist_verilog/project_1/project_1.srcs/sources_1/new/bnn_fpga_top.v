`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// BNN MNIST FPGA Top Module for Zynq-7000
//
// Features:
//   - Switch-based image selection (40 images: 4 per digit)
//   - Push button to trigger inference
//   - LED indicators for status and result
//   - ST7735 LCD output for image and digit display
//   - Seven-segment display output (optional, directly from PL)
//
// Interface:
//   - 6 switches: select image (sw[5:4]=variant, sw[3:0]=digit)
//   - 2 push buttons: btn[0]=run inference, btn[1]=update LCD
//   - 8 LEDs: status and digit output
//   - SPI to ST7735 LCD
//
// Compatible with: Xilinx Vivado 2023.2, Zynq-7000 (xc7z020clg484-1)
//////////////////////////////////////////////////////////////////////////////////

module bnn_fpga_top (
    // System
    input  wire         clk,            // 100MHz from PS or PL clock
    input  wire         rst_n,          // Active-low reset (directly from PL button)

    // User Interface
    input  wire [5:0]   sw,             // Slide switches for image selection
    input  wire [1:0]   btn,            // Push buttons (active high after debounce)
    output wire [7:0]   led,            // Status LEDs

    // LCD Interface (directly from PL)
    output wire         lcd_cs_n,       // Directly to Pmod connector
    output wire         lcd_dc,
    output wire         lcd_sclk,
    output wire         lcd_mosi,
    output wire         lcd_rst_n,

    // Seven Segment Display (directly from PL, active low segments)
    output wire [6:0]   seg,            // Directly to Pmod or discrete LEDs
    output wire [3:0]   an              // Anode control
);

    // -------------------------------------------------------------------------
    // Internal Signals
    // -------------------------------------------------------------------------
    wire [783:0] selected_image;
    wire [3:0]   expected_digit;
    wire         image_ready;

    wire [3:0]   predicted_digit;
    wire         bnn_valid;
    wire         bnn_busy;

    wire         lcd_busy;

    // -------------------------------------------------------------------------
    // Button Debounce
    // -------------------------------------------------------------------------
    reg [19:0] btn_cnt [0:1];
    reg [1:0]  btn_debounced;
    reg [1:0]  btn_prev;
    wire [1:0] btn_pressed;

    always @(posedge clk) begin
        if (!rst_n) begin
            btn_cnt[0] <= 20'd0;
            btn_cnt[1] <= 20'd0;
            btn_debounced <= 2'b0;
        end else begin
            // Button 0
            if (btn[0] != btn_debounced[0]) begin
                if (btn_cnt[0] < 20'd1_000_000)  // 10ms at 100MHz
                    btn_cnt[0] <= btn_cnt[0] + 20'd1;
                else begin
                    btn_cnt[0] <= 20'd0;
                    btn_debounced[0] <= btn[0];
                end
            end else begin
                btn_cnt[0] <= 20'd0;
            end

            // Button 1
            if (btn[1] != btn_debounced[1]) begin
                if (btn_cnt[1] < 20'd1_000_000)
                    btn_cnt[1] <= btn_cnt[1] + 20'd1;
                else begin
                    btn_cnt[1] <= 20'd0;
                    btn_debounced[1] <= btn[1];
                end
            end else begin
                btn_cnt[1] <= 20'd0;
            end
        end
    end

    always @(posedge clk) begin
        if (!rst_n)
            btn_prev <= 2'b0;
        else
            btn_prev <= btn_debounced;
    end

    // Rising edge detection
    assign btn_pressed = btn_debounced & ~btn_prev;

    // -------------------------------------------------------------------------
    // Control FSM
    // -------------------------------------------------------------------------
    localparam [2:0]
        CTRL_IDLE       = 3'd0,
        CTRL_LOAD_IMG   = 3'd1,
        CTRL_WAIT_IMG   = 3'd2,
        CTRL_RUN_BNN    = 3'd3,
        CTRL_WAIT_BNN   = 3'd4,
        CTRL_UPDATE_LCD = 3'd5,
        CTRL_WAIT_LCD   = 3'd6;

    reg [2:0]  ctrl_state;
    reg        load_image;
    reg        start_bnn;
    reg        update_lcd;
    reg [3:0]  result_digit;
    reg        result_valid;
    reg        result_correct;

    always @(posedge clk) begin
        if (!rst_n) begin
            ctrl_state   <= CTRL_IDLE;
            load_image   <= 1'b0;
            start_bnn    <= 1'b0;
            update_lcd   <= 1'b0;
            result_digit <= 4'd0;
            result_valid <= 1'b0;
            result_correct <= 1'b0;
        end else begin
            load_image <= 1'b0;
            start_bnn  <= 1'b0;
            update_lcd <= 1'b0;

            case (ctrl_state)

                CTRL_IDLE: begin
                    if (btn_pressed[0]) begin
                        // Run inference button pressed
                        ctrl_state <= CTRL_LOAD_IMG;
                        load_image <= 1'b1;
                    end else if (btn_pressed[1] && result_valid) begin
                        // Update LCD button pressed (only if we have a result)
                        ctrl_state <= CTRL_UPDATE_LCD;
                        update_lcd <= 1'b1;
                    end
                end

                CTRL_LOAD_IMG: begin
                    ctrl_state <= CTRL_WAIT_IMG;
                end

                CTRL_WAIT_IMG: begin
                    if (image_ready) begin
                        ctrl_state <= CTRL_RUN_BNN;
                        start_bnn  <= 1'b1;
                    end
                end

                CTRL_RUN_BNN: begin
                    ctrl_state <= CTRL_WAIT_BNN;
                end

                CTRL_WAIT_BNN: begin
                    if (bnn_valid) begin
                        result_digit   <= predicted_digit;
                        result_valid   <= 1'b1;
                        result_correct <= (predicted_digit == expected_digit);
                        ctrl_state     <= CTRL_UPDATE_LCD;
                        update_lcd     <= 1'b1;
                    end
                end

                CTRL_UPDATE_LCD: begin
                    ctrl_state <= CTRL_WAIT_LCD;
                end

                CTRL_WAIT_LCD: begin
                    if (!lcd_busy) begin
                        ctrl_state <= CTRL_IDLE;
                    end
                end

            endcase
        end
    end

    // -------------------------------------------------------------------------
    // Image Selector Instance
    // -------------------------------------------------------------------------
    image_selector u_img_sel (
        .clk            (clk),
        .rst_n          (rst_n),
        .sw             (sw),
        .load           (load_image),
        .image_out      (selected_image),
        .expected_digit (expected_digit),
        .ready          (image_ready)
    );

    // -------------------------------------------------------------------------
    // BNN Inference Core Instance
    // -------------------------------------------------------------------------
    bnn_core u_bnn (
        .clk        (clk),
        .rst_n      (rst_n),
        .start      (start_bnn),
        .image_in   (selected_image),
        .digit_out  (predicted_digit),
        .valid      (bnn_valid),
        .busy       (bnn_busy)
    );

    // -------------------------------------------------------------------------
    // LCD Controller Instance
    // -------------------------------------------------------------------------
    lcd_controller u_lcd (
        .clk         (clk),
        .rst_n       (rst_n),
        .update      (update_lcd),
        .image_in    (selected_image),
        .digit_in    (result_digit),
        .expected_in (expected_digit),
        .busy        (lcd_busy),
        .lcd_cs_n    (lcd_cs_n),
        .lcd_dc      (lcd_dc),
        .lcd_sclk    (lcd_sclk),
        .lcd_mosi    (lcd_mosi),
        .lcd_rst_n   (lcd_rst_n)
    );

    // -------------------------------------------------------------------------
    // LED Output
    // -------------------------------------------------------------------------
    // led[7]   = Correct prediction (green LED if available)
    // led[6]   = Result valid
    // led[5]   = BNN busy
    // led[4]   = LCD busy
    // led[3:0] = Predicted digit (binary)

    assign led[7]   = result_correct;
    assign led[6]   = result_valid;
    assign led[5]   = bnn_busy;
    assign led[4]   = lcd_busy;
    assign led[3:0] = result_digit;

    // -------------------------------------------------------------------------
    // Seven Segment Display (directly from PL, active low)
    // -------------------------------------------------------------------------
    reg [6:0] seg_data;
    reg [3:0] an_data;

    // Seven segment decoder (active low)
    always @(*) begin
        case (result_digit)
            4'd0: seg_data = 7'b1000000;
            4'd1: seg_data = 7'b1111001;
            4'd2: seg_data = 7'b0100100;
            4'd3: seg_data = 7'b0110000;
            4'd4: seg_data = 7'b0011001;
            4'd5: seg_data = 7'b0010010;
            4'd6: seg_data = 7'b0000010;
            4'd7: seg_data = 7'b1111000;
            4'd8: seg_data = 7'b0000000;
            4'd9: seg_data = 7'b0010000;
            default: seg_data = 7'b1111111;
        endcase
    end

    // Enable only first digit when result is valid
    assign an_data = result_valid ? 4'b1110 : 4'b1111;

    assign seg = seg_data;
    assign an  = an_data;

endmodule
