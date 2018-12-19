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
        url: "https://www.google.com",
        type: "GET",
        dataType: 'JSON',
        success: function (result) {
            result["gmv_redeem_xidian"] = parseInt(result["gmv_redeem_amount"] * 0.35);
            result["gmv_total_amount"] = parseInt(
                result["gmv_album_amount"] +
                result["gmv_vip_amount"] +
                result["gmv_recharge_remain_amount"] +
                result["gmv_redeem_amount"]
            );
            result["gmv_total_xidian"] = parseInt(
                result["gmv_album_xidian"] +
                result["gmv_vip_amount"] +
                result["gmv_recharge_remain_amount"] +
                result["gmv_redeem_xidian"]
            );


            result["gmv_redeem_xidian_yesterday"] = parseInt(result["gmv_redeem_yesterday"] * 0.35);
            result["gmv_total_yesterday"] = parseInt(
                result["gmv_album_yesterday"] +
                result["gmv_vip_yesterday"] +
                result["gmv_recharge_remain_yesterday"] +
                result["gmv_redeem_yesterday"]
            );
            result["gmv_total_xidian_yesterday"] = parseInt(
                result["gmv_album_xidian_yesterday"] +
                result["gmv_vip_yesterday"] +
                result["gmv_recharge_remain_yesterday"] +
                result["gmv_redeem_xidian_yesterday"]
            );


            result["gmv_album_rate"] = parseInt((result["gmv_album_amount"] / result["gmv_album_yesterday"] - 1) * 100);
            result["gmv_album_xidian_rate"] = parseInt((result["gmv_album_xidian"] / result["gmv_album_xidian_yesterday"] - 1) * 100);
            result["gmv_redeem_rate"] = parseInt((result["gmv_redeem_amount"] / result["gmv_redeem_yesterday"] - 1) * 100);
            result["gmv_redeem_xidian_rate"] = parseInt((result["gmv_redeem_xidian"] / result["gmv_redeem_xidian_yesterday"] - 1) * 100);
            result["gmv_vip_rate"] = parseInt((result["gmv_vip_amount"] / result["gmv_vip_yesterday"] - 1) * 100);
            result["gmv_recharge_remain_rate"] = parseInt((result["gmv_recharge_remain_amount"] / result["gmv_recharge_remain_yesterday"] - 1) * 100);
            result["gmv_total_rate"] = parseInt((result["gmv_total_amount"] / result["gmv_total_yesterday"] - 1) * 100);
            result["gmv_total_xidian_rate"] = parseInt((result["gmv_total_xidian"] / result["gmv_total_xidian_yesterday"] - 1) * 100);
            result["gmv_recharge_rate"] = parseInt((result["gmv_recharge"] / result["gmv_recharge_yesterday"] - 1) * 100);


            result["yesterday_gmv_redeem_xidian"] = parseInt(result["yesterday_gmv_redeem_amount"] * 0.35);
            result["yesterday_gmv_total"] = parseInt(
                result["yesterday_gmv_album_amount"] +
                result["yesterday_gmv_vip_amount"] +
                result["yesterday_gmv_recharge_remain_amount"] +
                result["yesterday_gmv_redeem_amount"]
            );
            result["yesterday_gmv_total_xidian"] = parseInt(
                result["yesterday_gmv_album_xidian"] +
                result["yesterday_gmv_vip_amount"] +
                result["yesterday_gmv_recharge_remain_amount"] +
                result["yesterday_gmv_redeem_xidian"]
            );
            var rateKeys = ["gmv_album_rate", "gmv_album_xidian_rate", "gmv_redeem_rate", "gmv_redeem_xidian_rate",
                "gmv_vip_rate", "gmv_recharge_remain_rate", "gmv_total_rate", "gmv_total_xidian_rate", "gmv_recharge_rate"];

            for (var key in result) {
                if ($.inArray(key, rateKeys) > "-1") {
                    if (result[key] > 0) {
                        var span = $("<span></span>");
                        var rate = $("." + key);
                        rate.empty();
                        rate.text(" +" + result[key] + "%");
                        rate.css({
                            "color": "red",
                            "font-size":"small"
                        });
                        rate.prepend(span);
                    }
                    else {
                        var span = $("<span></span>");
                        var rate = $("." + key);
                        rate.empty();
                        rate.text(" " + result[key] + "%");
                        rate.css({
                            "color": "green",
                            "font-size":"small"
                        });
                        rate.prepend(span);
                    }

                }
                else {
                    $("." + key).text(formatNum(result[key]));
                    $("." + key).css("font-size","medium")
                }
            }


        }
    });
}
showCurrentTime();
setInterval(showCurrentTime, 1000);
loadData();
setInterval(loadData, 5000);
