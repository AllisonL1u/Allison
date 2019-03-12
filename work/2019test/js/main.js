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
// 	$(".nano_1").text(formatNum(999999999999));
// 	$(".nano_1").css("font-size","small");
// 	$(".nano_2").text(formatNum(999999999999));
// 	$(".nano_2").css("font-size","small");
// 	$(".nano_3").text(formatNum(999999999999));
// 	$(".nano_3").css("font-size","small");
// 	$(".nano_4").text(formatNum(999999999999));
// 	$(".nano_4").css("font-size","small");
// 	$(".nano_5").text(formatNum(999999999999));
// 	$(".nano_5").css("font-size","small");
// 	$(".15_days_vip_free").text(formatNum(999999999999));
// 	$(".15_days_vip_free").css("font-size","small");
// 	$(".vip_purchase").text(formatNum(999999999999));
// 	$(".vip_purchase").css("font-size","small");
// 	$(".vip_gmv").text(formatNum(999999999999));
// 	$(".vip_gmv").css("font-size","small");
// 	$(".album_purchase_gmv").text(formatNum(999999999999));
// 	$(".album_purchase_gmv").css("font-size","small");
// 	$(".kol_course_pv").text(formatNum(999999999999));
// 	$(".kol_course_pv").css("font-size","small");
// 	$(".kol_course_uv").text(formatNum(999999999999));
// 	$(".kol_course_uv").css("font-size","small");
// 	$(".lunch_pv").text(formatNum(999999999999));
// 	$(".lunch_pv").css("font-size","small");
// 	$(".lunch_uv").text(formatNum(999999999999));
// 	$(".lunch_uv").css("font-size","small");
}

showCurrentTime();
setInterval(showCurrentTime, 1000);
loadData();
setInterval(loadData, 5000);

