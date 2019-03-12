/**
 * Created by Allison on 2018/12/19.
 */
 function formatNum(money) {
    for (var a = ""; money > 999;) {
        var s = money.toString();
        a = "," + s.substring(s.length - 3) + a;
        money = parseInt(money / 1000)
    }

    return money + a;
}

function showCurrentTime() {
    var time = new Date;
    var time = time.toLocaleDateString() + time.toLocaleTimeString();
    document.getElementById("time").innerHTML = time
}

function loadData() {
	$.ajax({
		url: "http://cn.bing.com",
        type: "GET",
        dataType: 'JSON',
        success: function (result) {
            	result['final_nano_weikuan_white'] = parseInt(result['final_nano_weikuan_1_white'] ? result['final_nano_weikuan_1_white'] : 0) + parseInt(result['final_nano_weikuan_2_white'] ? result['final_nano_weikuan_2_white'] : 0);
            	result['final_nano_weikuan_pink'] = parseInt(result['final_nano_weikuan_1_pink'] ? result['final_nano_weikuan_1_pink'] : 0) + parseInt(result['final_nano_weikuan_2_pink'] ? result['final_nano_weikuan_2_pink'] : 0);
            	result['final_nano_weikuan_blue'] = parseInt(result['final_nano_weikuan_1_blue'] ? result['final_nano_weikuan_1_blue'] : 0) + parseInt(result['final_nano_weikuan_2_blue'] ? result['final_nano_weikuan_2_blue'] : 0);
            	result['final_nano_weikuan_orange'] = parseInt(result['final_nano_weikuan_1_orange'] ? result['final_nano_weikuan_1_orange'] : 0) + parseInt(result['final_nano_weikuan_2_orange'] ? result['final_nano_weikuan_2_orange'] : 0);
            	result['final_pre_nano_count_1'] = parseInt(result['final_nano_weikuan_1_orange'] ? result['final_nano_weikuan_1_orange'] : 0) + parseInt(result['final_nano_weikuan_1_blue'] ? result['final_nano_weikuan_1_blue'] : 0) + parseInt(result['final_nano_weikuan_1_pink'] ? result['final_nano_weikuan_1_pink'] : 0) + parseInt(result['final_nano_weikuan_1_white'] ? result['final_nano_weikuan_1_white'] : 0);
            	result['final_pre_nano_count_2'] = parseInt(result['final_nano_weikuan_2_orange'] ? result['final_nano_weikuan_2_orange'] : 0) + parseInt(result['final_nano_weikuan_2_blue'] ? result['final_nano_weikuan_2_blue'] : 0) + parseInt(result['final_nano_weikuan_2_pink'] ? result['final_nano_weikuan_2_pink'] : 0) + parseInt(result['final_nano_weikuan_2_white'] ? result['final_nano_weikuan_2_white'] : 0);
            	result['final_pre_nano_count'] = result['final_pre_nano_count_1'] + result['final_pre_nano_count_2'];
        	for (var key in result) {
        		$("." + key).text(formatNum(result[key]));
                $("." + key).css("font-size","small");
        	}
        }
	});
	$(".final_nano_weikuan_white").text(formatNum(999999999999));
	$(".final_nano_weikuan_white").css("font-size","small");
	$(".final_nano_weikuan_blue").text(formatNum(999999999999));
	$(".final_nano_weikuan_blue").css("font-size","small");
	$(".final_nano_weikuan_pink").text(formatNum(999999999999));
	$(".final_nano_weikuan_pink").css("font-size","small");
	$(".final_nano_weikuan_orange").text(formatNum(999999999999));
	$(".final_nano_weikuan_orange").css("font-size","small");
	$(".final_nano_weikuan_1_white").text(formatNum(999999999999));
	$(".final_nano_weikuan_1_white").css("font-size","small");
	$(".final_nano_weikuan_1_pink").text(formatNum(999999999999));
	$(".final_nano_weikuan_1_pink").css("font-size","small");
	$(".final_nano_weikuan_1_orange").text(formatNum(999999999999));
	$(".final_nano_weikuan_1_orange").css("font-size","small");
	$(".final_nano_weikuan_1_blue").text(formatNum(999999999999));
	$(".final_nano_weikuan_1_blue").css("font-size","small");
	$(".final_nano_weikuan_2_orange").text(formatNum(999999999999));
	$(".final_nano_weikuan_2_orange").css("font-size","small");
	$(".final_nano_weikuan_2_blue").text(formatNum(999999999999));
	$(".final_nano_weikuan_2_blue").css("font-size","small");
	$(".final_nano_weikuan_2_pink").text(formatNum(999999999999));
	$(".final_nano_weikuan_2_pink").css("font-size","small");
	$(".final_nano_weikuan_2_white").text(formatNum(999999999999));
	$(".final_nano_weikuan_2_white").css("font-size","small");
	$(".final_pre_nano_count_1").text(formatNum(999999999999));
	$(".final_pre_nano_count_1").css("font-size","small");
	$(".final_pre_nano_count_2").text(formatNum(999999999999));
	$(".final_pre_nano_count_2").css("font-size","small");
	$(".final_pre_nano_count").text(formatNum(999999999999));
	$(".final_pre_nano_count").css("font-size","small");
}

showCurrentTime();
setInterval(showCurrentTime, 1000);
loadData();
setInterval(loadData, 5000);

