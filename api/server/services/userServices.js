const database = require('../src/models')
const Sequelize = require('sequelize');
const Op = Sequelize.Op;


class UserServices {

    // User Signup
    static async signup(newUser) {
        try {
            return await database.User.create(newUser);
        } catch (error){
            throw error;
        }
    }

    // find user with username or email
    static async findUser(data) {
        try {
            return await database.User.findOne({where: {[Op.or]: [{email: data.identifier}, {user_name: data.identifier}]}, include: [database.user_passwords]});
        } catch (error) {
            throw error
        }
    }

    // update user Information
    static async updateUser(data) {
        try {
            return await database.User.update(data, {where: {id: data.userId},
                returning: true,
                plain: true});
        } catch (error) {
            throw error
        }
     }

}

module.exports = UserServices;